"""Torch CUDA/MPS/CPU backend for native spike-train EGGROLL.

No gradients or surrogate gradients are used. ALL candidate forward passes,
LIF state updates, trace/count objectives, low-rank factors, estimated weight
gradients and Adam/SGD weight updates execute on the selected torch device.
Original numpy layer weights are synced only for external plotting/export.

Works with existing network_model.Layer.CMAES_FCLayer and EGGROLL_FCLayer.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from EGGROLL_utils import _check_compatible_layer, _layer_output_size


def select_device(device="auto") -> torch.device:
    """CUDA -> Apple Metal (MPS) -> CPU; explicit unavailable device raises."""
    if device is None or str(device).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available in this PyTorch installation")
    if selected.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Requested Apple MPS, but it is not available in this PyTorch installation")
    if selected.type not in ("cuda", "mps", "cpu"):
        raise ValueError("EGGROLL supports cuda, mps, cpu, or auto")
    return selected


def choose_dtype(dtype="float32", device="cpu"):
    result = {"float32": torch.float32, "float64": torch.float64,
              "double": torch.float64, "float": torch.float32}.get(str(dtype).lower())
    if result is None:
        raise ValueError("dtype must be 'float32' or 'float64'")
    if torch.device(device).type == "mps" and result == torch.float64:
        raise ValueError("Apple MPS does not support float64; use dtype='float32'")
    return result


def device_description(device: torch.device) -> str:
    if device.type == "cuda":
        return f"CUDA GPU ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU (PyTorch fallback)"


@dataclass
class Perturbation:
    a_pos: torch.Tensor             # (pairs,in,rank)
    b_pos: torch.Tensor             # (pairs,out,rank)
    a_full: torch.Tensor            # (P,in,rank), antithetic interleaved
    b_full: torch.Tensor            # (P,out,rank), negated on odd candidates


@dataclass
class PreparedSample:
    spikes: torch.Tensor           # (T,input)
    targets: torch.Tensor          # (T,output)
    target_trace: torch.Tensor     # (T,output) full causal trace BEFORE cropping


class TorchSpikingModel:
    """GPU-resident view of existing numpy LIF layers, including frozen layers."""

    def __init__(self, net, device="auto", dtype="float32"):
        self.net = net
        self.device = select_device(device)
        self.dtype = choose_dtype(dtype, self.device)
        self.weights: dict[int, torch.Tensor] = {}
        self.thresholds: dict[int, torch.Tensor] = {}
        self.decays: dict[int, float] = {}
        for i, layer in enumerate(net.layers):
            self.register(layer)
            print(f"Initialised layer {i}")

    def register(self, layer):
        _check_compatible_layer(layer)
        key = id(layer)
        self.weights[key] = torch.as_tensor(np.asarray(layer.weights),
                                            dtype=self.dtype, device=self.device).clone()
        self.thresholds[key] = torch.as_tensor(np.asarray(layer.threshold),
                                               dtype=self.dtype, device=self.device)
        self.decays[key] = float(layer.decay)

    def unregister(self, layer):
        for mapping in (self.weights, self.thresholds, self.decays):
            mapping.pop(id(layer), None)

    @torch.no_grad()
    def sync_to_numpy(self, modules=None):
        """Copy device-resident weights back to existing layers only when needed."""
        for layer in (self.net.layers if modules is None else modules):
            layer.weights = self.weights[id(layer)].detach().cpu().numpy().astype(np.float64, copy=True)

    @torch.no_grad()
    def snapshot(self, modules):
        return [self.weights[id(layer)].clone() for layer in modules]

    @torch.no_grad()
    def restore(self, modules, snapshot):
        for layer, weight in zip(modules, snapshot):
            self.weights[id(layer)].copy_(weight)

    def _spikes_to_device(self, batch):
        spikes = batch[0] if isinstance(batch, (tuple, list)) else batch
        if isinstance(spikes, np.ndarray):
            # torch.as_tensor on numpy avoids an extra intermediate copy.
            x = torch.as_tensor(spikes)
        elif isinstance(spikes, torch.Tensor):
            x = spikes.detach()
        else:
            x = torch.as_tensor(spikes)
        while x.ndim > 2 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 2:
            raise ValueError(f"Expected one spike train (T,N), batch_size=1, got {tuple(x.shape)}")
        input_num = int(self.net.layers[0].input_num)
        if x.shape[1] != input_num:
            if x.shape[0] == input_num:
                x = x.transpose(0, 1)
            else:
                raise ValueError(f"Spike train {tuple(x.shape)} does not match input width {input_num}")
        return x.to(device=self.device, dtype=self.dtype).contiguous()

    @torch.no_grad()
    def _step(self, x, layer, u, factors=None, sigma=0.0):
        """Population x=(P,input), u=(P,output); keep ALL operations on GPU."""
        w = self.weights[id(layer)]
        current = x @ w  # shared base weight GEMM, not P separate dense matrices
        if factors is not None:
            # (P,input,rank) and (P,output,rank): no dense candidate delta-W.
            a, b = factors.a_full, factors.b_full
            projection = torch.einsum("pi,pir->pr", x, a)
            current = current + (sigma / (a.shape[-1] ** .5)) * torch.einsum("pr,por->po", projection, b)
        u = u + current
        fired = u >= self.thresholds[id(layer)]
        spikes = fired.to(self.dtype)
        u = torch.where(fired, torch.zeros_like(u), u).clamp_min(0.) * self.decays[id(layer)]
        return spikes, u

    @torch.no_grad()
    def _stage_target(self, spikes, stage):
        if stage is None or stage.target_mode == "global_input":
            return spikes
        first = min(stage.train_indices)
        frozen = self.net.layers[:first]
        if not frozen:
            return spikes
        states = [torch.zeros((1, _layer_output_size(layer)), device=self.device, dtype=self.dtype)
                  for layer in frozen]
        targets = []
        for sample_t in spikes:
            x = sample_t.unsqueeze(0)
            for i, layer in enumerate(frozen):
                x, states[i] = self._step(x, layer, states[i])
            targets.append(x.squeeze(0))
        return torch.stack(targets)

    @torch.no_grad()
    def prepare(self, batches, stage=None, trace_alpha=.9):
        """Copy spike samples to GPU once; cache stage targets & causal traces."""
        if not 0 <= trace_alpha <= 1:
            raise ValueError("trace_alpha must be between 0 and 1")
        prepared = []
        for batch in batches:
            spikes = self._spikes_to_device(batch)
            targets = self._stage_target(spikes, stage)
            state = torch.zeros(targets.shape[1], device=self.device, dtype=self.dtype)
            traces = []
            for target_t in targets:
                state = trace_alpha * state + target_t
                traces.append(state)
            prepared.append(PreparedSample(spikes, targets, torch.stack(traces)))
        if not prepared:
            raise ValueError("Empty batch list")
        return prepared

    @torch.no_grad()
    def evaluate(self, samples, population_size=1, noise=None, sigma=0.,
                 trace_alpha=.9, count_weight=.01, delay=0,
                 stage=None, temp_readout=None):
        """Returns population loss tensor on selected device (even P=1).

        Streaming over time: no (time,population,output) prediction tensor.
        Fixed target trace is computed once in prepare().
        """
        active_layers = (self.net.layers if stage is None else self.net.layers[:stage.active_end + 1])
        if temp_readout is not None:
            active_layers = list(active_layers) + [temp_readout]
        losses = torch.zeros(population_size, device=self.device, dtype=self.dtype)
        for sample in samples:
            spikes, targets, target_trace = sample.spikes, sample.targets, sample.target_trace
            T, output_size = targets.shape
            if abs(delay) >= T:
                raise ValueError(f"abs(delay) must be < spike-train length {T}")
            if _layer_output_size(active_layers[-1]) != output_size:
                raise ValueError(f"Output width {_layer_output_size(active_layers[-1])} != target width {output_size}")
            states = [torch.zeros((population_size, _layer_output_size(layer)),
                                  dtype=self.dtype, device=self.device) for layer in active_layers]
            output_trace = torch.zeros((population_size, output_size), dtype=self.dtype, device=self.device)
            trace_sse = torch.zeros(population_size, dtype=self.dtype, device=self.device)
            pred_count = torch.zeros(population_size, dtype=self.dtype, device=self.device)
            target_count = targets[max(0, -delay): min(T, T-delay)].sum()
            for t in range(T):
                x = spikes[t].unsqueeze(0).expand(population_size, -1)
                for i, layer in enumerate(active_layers):
                    factors = None if noise is None else noise.get(id(layer))
                    x, states[i] = self._step(x, layer, states[i], factors, sigma)
                output_trace = trace_alpha * output_trace + x
                target_index = t - delay
                if 0 <= target_index < T:
                    trace_sse = trace_sse + ((output_trace - target_trace[target_index]) ** 2).mean(dim=1)
                    pred_count = pred_count + x.sum(dim=1)
            trace_loss = trace_sse / (T - abs(delay))
            count_loss = ((pred_count - target_count) / (target_count + 1e-12)) ** 2
            losses += trace_loss + count_weight * count_loss
        return losses / len(samples)

    @torch.no_grad()
    def base_loss(self, samples, **kwargs):
        return float(self.evaluate(samples, population_size=1, **kwargs)[0].item())


class TorchEGGROLL:
    """Antithetic rank-r ES and full-weight Adam/SGD updates entirely on device."""

    def __init__(self, backend: TorchSpikingModel, modules, sigma=.15, learning_rate=.02,
                 population_size=32, rank=4, seed=None, fitness_shaping="centered_rank",
                 optimizer="adam", weight_decay=0., grad_clip=None):
        self.backend = backend
        self.modules = list(modules)
        if not self.modules:
            raise ValueError("At least one trainable layer required")
        if sigma <= 0 or learning_rate <= 0 or population_size < 2 or population_size % 2 or rank < 1:
            raise ValueError("sigma/lr > 0, population_size even and >=2, rank >=1 are required")
        if fitness_shaping not in ("centered_rank", "normalized", "raw"):
            raise ValueError("fitness_shaping must be centered_rank, normalized, or raw")
        if optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be adam or sgd")
        for layer in self.modules:
            if id(layer) not in backend.weights:
                raise ValueError("Trainable layer not registered on backend")
        self.sigma, self.learning_rate = float(sigma), float(learning_rate)
        self.population_size, self.pairs, self.rank = int(population_size), int(population_size//2), int(rank)
        self.fitness_shaping, self.optimizer = fitness_shaping, optimizer
        self.weight_decay, self.grad_clip = float(weight_decay), grad_clip
        # MPS random generators are unsupported or inconsistent across builds.
        # Only the small random factors need transfer; forward/update stay on GPU.
        if backend.device.type == "mps":
            self.generator = torch.Generator(device="cpu")
            self.noise_device = torch.device("cpu")
        else:
            self.generator = torch.Generator(device=backend.device)
            self.noise_device = backend.device
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.t = 0
        self.m = [torch.zeros_like(backend.weights[id(layer)]) for layer in self.modules]
        self.v = [torch.zeros_like(backend.weights[id(layer)]) for layer in self.modules]
        print(f"Initialised model with sigma = {self.sigma}, learning rate = {self.learning_rate}")

    @torch.no_grad()
    def ask(self):
        perturbations = {}
        for layer in self.modules:
            ni, no = self.backend.weights[id(layer)].shape
            a = torch.randn((self.pairs, ni, self.rank), generator=self.generator,
                            dtype=self.backend.dtype, device=self.noise_device).to(self.backend.device)
            b = torch.randn((self.pairs, no, self.rank), generator=self.generator,
                            dtype=self.backend.dtype, device=self.noise_device).to(self.backend.device)
            aa = torch.repeat_interleave(a, 2, dim=0)
            bb = torch.repeat_interleave(b, 2, dim=0)
            bb[1::2] = -bb[1::2]
            perturbations[id(layer)] = Perturbation(a, b, aa, bb)
        return perturbations

    @staticmethod
    def _fitness(losses, method):
        """Tie-aware centered ranks: identical losses produce identical utility."""
        losses = np.asarray(losses, dtype=np.float64)
        if not np.all(np.isfinite(losses)):
            raise FloatingPointError("Non-finite population loss")
        if method == "raw":
            return losses - losses.mean()
        if method == "normalized":
            std = losses.std()
            return (losses - losses.mean()) / std if std > 1e-12 else np.zeros_like(losses)
        order = np.argsort(losses, kind="stable")
        ordered = losses[order]
        rank_scores = np.empty_like(losses)
        start = 0
        while start < len(losses):
            end = start + 1
            while end < len(losses) and ordered[end] == ordered[start]:
                end += 1
            rank_scores[order[start:end]] = (start + end - 1) * .5
            start = end
        return rank_scores / max(len(losses)-1, 1) - .5

    @torch.no_grad()
    def tell(self, noise, losses):
        if losses.shape != (self.population_size,):
            raise ValueError(f"Expected {self.population_size} losses, got {tuple(losses.shape)}")
        # Only P scalar losses copied CPU for tie-aware fitness shaping/logging;
        # all big tensors and update computation stay on the accelerated device.
        loss_np = losses.detach().cpu().numpy().astype(np.float64)
        shaped = self._fitness(loss_np, self.fitness_shaping)
        utility = torch.as_tensor(shaped, dtype=self.backend.dtype, device=self.backend.device)
        diff = utility[0::2] - utility[1::2]
        gradients = []
        for layer in self.modules:
            p = noise[id(layer)]
            grad = torch.einsum("p,pir,por->io", diff, p.a_pos, p.b_pos)
            grad = grad / (2 * self.pairs * self.sigma * self.rank ** .5)
            if self.weight_decay:
                grad += self.weight_decay * self.backend.weights[id(layer)]
            gradients.append(grad)
        norm = torch.sqrt(sum((g*g).sum() for g in gradients))
        grad_norm = float(norm.item())
        if self.grad_clip is not None and grad_norm > float(self.grad_clip):
            gradients = [g * (float(self.grad_clip) / grad_norm) for g in gradients]
        self.t += 1
        for i, (layer, grad) in enumerate(zip(self.modules, gradients)):
            if self.optimizer == "adam":
                self.m[i].mul_(.9).add_(grad, alpha=.1)
                self.v[i].mul_(.999).addcmul_(grad, grad, value=.001)
                step = (self.m[i] / (1 - .9 ** self.t)) / (
                    torch.sqrt(self.v[i] / (1 - .999 ** self.t)) + 1e-8)
            else:
                step = grad
            self.backend.weights[id(layer)].add_(step, alpha=-self.learning_rate)
        return {"gradient_norm": grad_norm, "best_candidate_loss": float(loss_np.min()),
                "mean_candidate_loss": float(loss_np.mean())}
