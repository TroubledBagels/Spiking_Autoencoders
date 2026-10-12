"""Shared NumPy EGGROLL utilities for the existing spiking autoencoder models.

Implements paired, rank-r perturbations AB.T / sqrt(rank), vectorised population
inference using the *shared* base weights, and Adam/SGD updates of full matrices.
No autograd, cma dependency, or synthetic spike encoding is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import gc
import numpy as np


# =============================================================================
# Existing network, data, reconstruction-loss, and metric conventions
# =============================================================================

def _layer_input_size(layer):
    return int(layer.input_num)


def _layer_output_size(layer):
    if hasattr(layer, "layer_size"):
        return int(layer.layer_size)
    for attr in ("output_num", "output_size", "out_features"):
        if hasattr(layer, attr):
            return int(getattr(layer, attr))
    w = np.asarray(layer.weights)
    if w.ndim != 2 or w.shape[0] != _layer_input_size(layer):
        raise ValueError("Expected weights with shape (input_num, layer_size)")
    return int(w.shape[1])


def _check_compatible_layer(layer):
    if not hasattr(layer, "weights") or not hasattr(layer, "input_num"):
        raise TypeError("EGGROLL requires fully connected CMAES_FCLayer/EGGROLL_FCLayer-style layers")
    shape = (_layer_input_size(layer), _layer_output_size(layer))
    if np.asarray(layer.weights).shape != shape:
        raise ValueError(f"{type(layer).__name__} weights must be {shape}")
    if not hasattr(layer, "u") or not hasattr(layer, "threshold") or not hasattr(layer, "decay"):
        raise TypeError("Only the supplied noncompetitive LIF CMAES/EGGROLL FC layers are supported")
    if hasattr(layer, "n_type") and getattr(layer.n_type, "name", "LIF_total") != "LIF_total":
        raise NotImplementedError("Only NeuronType.LIF_total is supported")


def get_module_parameters(modules):
    arrays = [np.asarray(m.weights, dtype=np.float64).ravel() for m in modules]
    if not arrays:
        raise ValueError("No trainable layers")
    return np.concatenate(arrays).copy()


def set_module_parameters(modules, theta):
    theta = np.asarray(theta, dtype=np.float64).ravel()
    offset = 0
    for m in modules:
        shape = np.asarray(m.weights).shape
        size = int(np.prod(shape))
        if offset + size > theta.size:
            raise ValueError("Parameter vector too short")
        m.weights = theta[offset:offset + size].reshape(shape).copy()
        offset += size
    if offset != theta.size:
        raise ValueError(f"Expected {offset} parameters, got {theta.size}")


def get_parameters(net):
    return get_module_parameters(net.layers)


def set_parameters(net, theta):
    return set_module_parameters(net.layers, theta)


def extract_spike_train(batch, input_num):
    spikes = batch[0] if isinstance(batch, (tuple, list)) else batch
    if hasattr(spikes, "detach"):
        spikes = spikes.detach().cpu().numpy()
    spikes = np.asarray(spikes)
    while spikes.ndim > 2 and spikes.shape[0] == 1:
        spikes = spikes[0]
    if spikes.ndim != 2:
        raise ValueError(f"Expected a single 2D spike raster (use DataLoader batch_size=1), got {spikes.shape}")
    if spikes.shape[1] == input_num:
        return spikes.astype(np.float64, copy=False)
    if spikes.shape[0] == input_num:
        return spikes.T.astype(np.float64, copy=False)
    raise ValueError(f"Spike shape {spikes.shape} does not match {input_num} inputs")


def collect_samples(loader, count):
    if count < 1:
        raise ValueError("samples_per_generation and test_samples must be >= 1")
    samples = []
    for batch in loader:
        samples.append(batch)
        if len(samples) >= count:
            break
    if not samples:
        raise ValueError("Data loader yielded no samples")
    return samples


def get_sample(loader, sample_index=0):
    for i, batch in enumerate(loader):
        if i == sample_index:
            return batch
    raise IndexError(f"Sample index {sample_index} outside dataset")


def run_autoencoder(net, spike_train):
    net.reset_state()
    latent_index = max(len(net.layers) // 2 - 1, 0)
    latent, output = [], []
    for spikes_t in spike_train:
        x = spikes_t
        for index, layer in enumerate(net.layers):
            x = layer(x)
            if index == latent_index:
                latent_t = np.asarray(x).copy()
        latent.append(latent_t)
        output.append(np.asarray(x).copy())
    return np.asarray(latent), np.asarray(output)


def spike_trace(spikes, alpha=0.9):
    spikes = np.asarray(spikes, dtype=np.float64)
    if not 0 <= alpha <= 1:
        raise ValueError("trace_alpha must be between 0 and 1")
    traces = np.empty_like(spikes)
    state = np.zeros(spikes.shape[1], dtype=np.float64)
    for t in range(len(spikes)):
        state = alpha * state + spikes[t]
        traces[t] = state
    return traces


def delay_description(delay):
    if delay > 0:
        return f"reconstruct {delay} timestep(s) into the past"
    if delay < 0:
        return f"predict {-delay} timestep(s) into the future"
    return "reconstruct current timestep"


def align_for_delay(input_values, output_values, delay=0):
    a, b = np.asarray(input_values), np.asarray(output_values)
    if a.shape != b.shape:
        raise ValueError(f"Input {a.shape} does not match output {b.shape}")
    if abs(delay) >= len(a):
        raise ValueError(f"abs(delay) must be < sequence length {len(a)}")
    if delay > 0:
        return a[:-delay], b[delay:]
    if delay < 0:
        return a[-delay:], b[:delay]
    return a, b


def reconstruction_loss_components(input_spikes, output_spikes, trace_alpha=0.9, delay=0):
    target_trace, pred_trace = align_for_delay(
        spike_trace(input_spikes, trace_alpha), spike_trace(output_spikes, trace_alpha), delay
    )
    target_spikes, pred_spikes = align_for_delay(input_spikes, output_spikes, delay)
    trace_loss = np.mean((target_trace - pred_trace) ** 2)
    input_count = np.sum(target_spikes)
    output_count = np.sum(pred_spikes)
    count_loss = ((output_count - input_count) / (input_count + 1e-12)) ** 2
    return float(trace_loss), float(count_loss)


def reconstruction_loss(input_spikes, output_spikes, trace_alpha=0.9, count_weight=0.01, delay=0):
    tl, cl = reconstruction_loss_components(input_spikes, output_spikes, trace_alpha, delay)
    return tl + count_weight * cl


def reconstruction_counts(input_spikes, output_spikes, delay=0):
    a, b = align_for_delay(input_spikes, output_spikes, delay)
    a, b = a == 1, b == 1
    return int(np.sum(a & b)), int(np.sum(~a & b)), int(np.sum(a & ~b))


def reconstruction_counts_tolerant(input_spikes, output_spikes, tolerance=1, delay=0):
    a, b = align_for_delay(input_spikes, output_spikes, delay)
    tp, target_count, pred_count = 0, 0, 0
    for n in range(a.shape[1]):
        aa = np.flatnonzero(a[:, n] == 1)
        bb = np.flatnonzero(b[:, n] == 1)
        target_count += len(aa)
        pred_count += len(bb)
        i = j = 0
        while i < len(aa) and j < len(bb):
            if abs(aa[i] - bb[j]) <= tolerance:
                tp += 1
                i += 1
                j += 1
            elif bb[j] < aa[i] - tolerance:
                j += 1
            else:
                i += 1
    return int(tp), int(pred_count - tp), int(target_count - tp)


def calculate_prf(tp, fp, fn):
    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    return float(p), float(r), float(2 * p * r / (p + r + 1e-12))


def evaluate_candidate(net, theta, samples, trace_alpha=0.9, count_weight=0.01, delay=0):
    set_parameters(net, theta)
    return evaluate_base_loss(net, samples, trace_alpha=trace_alpha, count_weight=count_weight, delay=delay)


def evaluate_base_loss(net, samples, trace_alpha=0.9, count_weight=0.01, delay=0, stage=None, temp_readout=None):
    losses = []
    for batch in samples:
        spikes = extract_spike_train(batch, net.layers[0].input_num)
        if stage is None:
            _, outputs = run_autoencoder(net, spikes)
            target = spikes
        else:
            target, outputs = run_symmetric_stage(net, spikes, stage, temp_readout)
        losses.append(reconstruction_loss(target, outputs, trace_alpha, count_weight, delay))
    if not losses:
        raise ValueError("No evaluation samples")
    return float(np.mean(losses))


def silent_output_loss(spike_train, trace_alpha=0.9, count_weight=0.01, delay=0):
    return reconstruction_loss(spike_train, np.zeros_like(spike_train), trace_alpha, count_weight, delay)


def test_silent_baseline(net, test_loader, trace_alpha=0.9, count_weight=0.01, delay=0):
    ls = np.zeros(3)
    num = 0
    for batch in test_loader:
        spikes = extract_spike_train(batch, net.layers[0].input_num)
        tl, cl = reconstruction_loss_components(spikes, np.zeros_like(spikes), trace_alpha, delay)
        ls += (tl + count_weight * cl, tl, cl)
        num += 1
    if not num:
        raise ValueError("Empty test loader")
    return dict(zip(("loss", "trace_loss", "count_loss"), (ls / num).tolist()))


def evaluate_test_metrics(net, test_loader, trace_alpha=0.9, count_weight=0.01, delay=0,
                          stage=None, temp_readout=None):
    sums = dict(loss=0., trace_loss=0., count_loss=0., input_spikes=0., output_spikes=0.)
    counts = {tol: np.zeros(3, dtype=np.int64) for tol in (0, 1, 2)}
    num = 0
    for batch in test_loader:
        spikes = extract_spike_train(batch, net.layers[0].input_num)
        if stage is None:
            _, outputs = run_autoencoder(net, spikes)
            targets = spikes
        else:
            targets, outputs = run_symmetric_stage(net, spikes, stage, temp_readout)
        tl, cl = reconstruction_loss_components(targets, outputs, trace_alpha, delay)
        sums["loss"] += tl + count_weight * cl
        sums["trace_loss"] += tl
        sums["count_loss"] += cl
        a, b = align_for_delay(targets, outputs, delay)
        sums["input_spikes"] += float(a.sum())
        sums["output_spikes"] += float(b.sum())
        counts[0] += reconstruction_counts(targets, outputs, delay)
        counts[1] += reconstruction_counts_tolerant(targets, outputs, 1, delay)
        counts[2] += reconstruction_counts_tolerant(targets, outputs, 2, delay)
        num += 1
    if not num:
        raise ValueError("Empty test loader")
    for key in ("loss", "trace_loss", "count_loss"):
        sums[key] /= num
    for tol in (0, 1, 2):
        p, r, f = calculate_prf(*counts[tol])
        suffix = "" if tol == 0 else f"_{tol}"
        sums[f"precision{suffix}"] = p
        sums[f"recall{suffix}"] = r
        sums[f"f1{suffix}"] = f
    sums["tp"], sums["fp"], sums["fn"] = map(int, counts[0])
    sums["spike_ratio"] = sums["output_spikes"] / (sums["input_spikes"] + 1e-12)
    return sums


def plot_spike_raster(ax, spikes, title):
    for neuron in range(spikes.shape[1]):
        times = np.flatnonzero(spikes[:, neuron] == 1)
        if len(times):
            ax.scatter(times, np.full(len(times), neuron), marker="|", s=18)
    ax.set_title(title)
    ax.set_ylabel("Neuron")
    ax.set_ylim(-.5, spikes.shape[1] - .5)
    if spikes.shape[1] <= 50:
        ax.set_yticks(np.arange(spikes.shape[1]))
    ax.grid(axis="x", alpha=.2)


def _plot_reconstruction(targets, outputs, title, file_path, delay=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, b = align_for_delay(targets, outputs, delay)
    _, _, f = calculate_prf(*reconstruction_counts(a, b))
    _, _, f1 = calculate_prf(*reconstruction_counts_tolerant(a, b, 1))
    _, _, f2 = calculate_prf(*reconstruction_counts_tolerant(a, b, 2))
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)
    plot_spike_raster(axes[0], a, f"Target spikes | count={int(a.sum())}")
    plot_spike_raster(axes[1], b, f"Output spikes | count={int(b.sum())} | ratio={b.sum()/(a.sum()+1e-12):.3f}")
    axes[1].set_xlabel("Aligned timestep")
    fig.suptitle(f"{title} | F1={f:.4f} | +/-1 F1={f1:.4f} | +/-2 F1={f2:.4f}")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    del fig, axes
    gc.collect()


def plot_selected_sample(net, theta, spike_train, generation, plot_dir,
                         trace_alpha=0.9, count_weight=0.01, delay=0):
    set_parameters(net, theta)
    _, outputs = run_autoencoder(net, spike_train)
    loss = reconstruction_loss(spike_train, outputs, trace_alpha, count_weight, delay)
    _plot_reconstruction(spike_train, outputs,
                         f"Generation {generation:03d} | delay={delay} | loss={loss:.6f}",
                         Path(plot_dir) / f"generation_{generation:03d}.png", delay)
    return outputs


def export_weights(net, output_dir="ExportedWeights"):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(net.layers):
        if hasattr(layer, "weights"):
            output = path / f"layer_{i + 1}_weights.npy"
            np.save(output, layer.weights)
            print(f"Saved layer {i+1} weights to {output}")


# =============================================================================
# The EXACT stage sequencing / local-vs-global targets of SymmetricCMAES.py
# =============================================================================

@dataclass
class SymmetricStage:
    name: str
    structure: list
    active_end: int
    train_indices: list
    target_mode: str
    use_temp_readout: bool
    temp_input_size: int | None = None
    temp_output_size: int | None = None


def network_widths(net):
    if not net.layers:
        raise ValueError("Network has no layers")
    sizes = [_layer_input_size(net.layers[0])]
    for layer in net.layers:
        if _layer_input_size(layer) != sizes[-1]:
            raise ValueError("Disconnected layer widths")
        sizes.append(_layer_output_size(layer))
    return sizes


def build_symmetric_stages(net):
    widths = network_widths(net)
    count = len(net.layers)
    if count % 2 != 0 or count < 2:
        raise ValueError("Symmetric training expects an even number of synaptic layers")
    if widths != widths[::-1]:
        raise ValueError(f"Expected a symmetric autoencoder, got {widths}")
    depth = count // 2
    stages = []
    for i in range(depth-1):
        structure = widths[:i + 2] + [widths[i]]
        stages.append(SymmetricStage(" -> ".join(map(str, structure)), structure,
                                     i, [i], "local_input", True, widths[i+1], widths[i]))
    ec, dc = depth-1, depth
    structure = widths[:dc+2]
    stages.append(SymmetricStage(" -> ".join(map(str, structure)), structure,
                                 dc, [ec, dc], "global_input" if depth == 1 else "local_input", False))
    for i in range(1, depth-1):
        actual = depth + i
        structure = widths[:actual+2] + [widths[actual]]
        stages.append(SymmetricStage(" -> ".join(map(str, structure)), structure,
                                     actual, [actual], "local_input", True,
                                     widths[actual+1], widths[actual]))
    if depth > 1:
        stages.append(SymmetricStage(" -> ".join(map(str, widths)), widths,
                                     count-1, [count-1], "global_input", False))
    return stages


def describe_symmetric_schedule(stages):
    print("Symmetric progressive schedule:")
    for i, stage in enumerate(stages, 1):
        suffix = " + temporary readout" if stage.use_temp_readout else ""
        print(f"  Stage {i}: {stage.name} | optimize real layers {stage.train_indices}{suffix}")


def stage_trainable_modules(net, stage, temp_readout=None):
    modules = [net.layers[i] for i in stage.train_indices]
    if temp_readout is not None:
        modules.append(temp_readout)
    return modules


def run_symmetric_stage(net, spike_train, stage, temp_readout=None):
    net.reset_state()
    if temp_readout is not None:
        temp_readout.reset_state()
    targets, outputs = [], []
    first = min(stage.train_indices)
    for spikes_t in spike_train:
        x = np.asarray(spikes_t)
        target_t = x.copy() if stage.target_mode == "global_input" else None
        for i in range(stage.active_end + 1):
            if stage.target_mode == "local_input" and i == first:
                target_t = np.asarray(x).copy()
            x = net.layers[i](x)
        if temp_readout is not None:
            x = temp_readout(x)
        targets.append(target_t)
        outputs.append(np.asarray(x).copy())
    return np.asarray(targets), np.asarray(outputs)


def plot_stage_sample(net, stage, temp_readout, theta, batch, generation, stage_number,
                      plot_dir, trace_alpha=0.9, count_weight=0.01, delay=0):
    set_module_parameters(stage_trainable_modules(net, stage, temp_readout), theta)
    spikes = extract_spike_train(batch, net.layers[0].input_num)
    targets, outputs = run_symmetric_stage(net, spikes, stage, temp_readout)
    loss = reconstruction_loss(targets, outputs, trace_alpha, count_weight, delay)
    _plot_reconstruction(targets, outputs,
                         f"Stage {stage_number}: {stage.name} | gen {generation:03d} | loss={loss:.6f}",
                         Path(plot_dir) / f"stage_{stage_number:02d}" / f"generation_{generation:03d}.png", delay)


# =============================================================================
# EGGROLL: low-rank, antithetic, batched evaluation + full-matrix updates
# =============================================================================

@dataclass
class LayerPerturbation:
    # Each pair has two evaluations in order (+epsilon, -epsilon).
    a_pos: np.ndarray             # (pairs, input, rank)
    b_pos: np.ndarray             # (pairs, output, rank)
    a_full: np.ndarray            # (2*pairs, input, rank)
    b_full: np.ndarray            # (2*pairs, output, rank) ; second sign negated


class EGGROLL:
    """Rank-r antithetic evolution strategies with Adam or SGD full-weight updates.

    The matrix perturbation of one candidate is sigma * A B.T / sqrt(rank).
    The antithetic gradient estimator (minimization) is
         sum_i (loss_i+ - loss_i-) * (A_i @ B_i.T)
         -------------------------------------------------
                         2 * pairs * sigma * sqrt(rank)
    after optional fitness shaping. Full matrices are updated; no fixed LoRA.
    """
    def __init__(self, modules, sigma=0.15, learning_rate=0.02, population_size=32,
                 rank=4, seed=None, fitness_shaping="centered_rank", optimizer="adam",
                 weight_decay=0., grad_clip=None):
        self.modules = list(modules)
        if not self.modules:
            raise ValueError("At least one trainable module required")
        for m in self.modules:
            _check_compatible_layer(m)
        if sigma <= 0 or learning_rate <= 0 or rank < 1 or population_size < 2 or population_size % 2:
            raise ValueError("sigma and learning_rate must be > 0; rank >= 1; population_size even and >= 2")
        if fitness_shaping not in ("centered_rank", "normalized", "raw"):
            raise ValueError("fitness_shaping must be centered_rank, normalized, or raw")
        if optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be adam or sgd")
        self.sigma = float(sigma)
        self.learning_rate = float(learning_rate)
        self.population_size = int(population_size)
        self.pairs = population_size // 2
        self.rank = int(rank)
        self.rng = np.random.default_rng(seed)
        self.fitness_shaping = fitness_shaping
        self.optimizer = optimizer
        self.weight_decay = float(weight_decay)
        self.grad_clip = grad_clip
        self.t = 0
        self.m = [np.zeros_like(mod.weights, dtype=np.float64) for mod in self.modules]
        self.v = [np.zeros_like(mod.weights, dtype=np.float64) for mod in self.modules]

    def ask(self):
        perturbations = {}
        for mod in self.modules:
            n_in, n_out = np.asarray(mod.weights).shape
            a = self.rng.standard_normal((self.pairs, n_in, self.rank))
            b = self.rng.standard_normal((self.pairs, n_out, self.rank))
            a_full = np.repeat(a, 2, axis=0)
            b_full = np.repeat(b, 2, axis=0)
            b_full[1::2] *= -1.0
            perturbations[id(mod)] = LayerPerturbation(a, b, a_full, b_full)
        return perturbations

    @staticmethod
    def _fitness(losses, method):
        losses = np.asarray(losses, dtype=np.float64)
        if not np.all(np.isfinite(losses)):
            raise FloatingPointError("Encountered a non-finite population loss")
        if method == "raw":
            return losses - losses.mean()
        if method == "normalized":
            std = losses.std()
            return (losses - losses.mean()) / std if std > 1e-12 else np.zeros_like(losses)
        # Ties must have IDENTICAL utility (important for all-silent spike trains).
        order = np.argsort(losses, kind="stable")
        sorted_loss = losses[order]
        ranked = np.empty_like(losses)
        start = 0
        while start < len(losses):
            end = start + 1
            while end < len(losses) and sorted_loss[end] == sorted_loss[start]:
                end += 1
            ranked[order[start:end]] = (start + end - 1) / 2
            start = end
        return ranked / max(len(losses)-1, 1) - .5

    def tell(self, noise, losses):
        losses = np.asarray(losses, dtype=np.float64)
        if losses.shape != (self.population_size,):
            raise ValueError(f"Expected {self.population_size} losses, got {losses.shape}")
        utility = self._fitness(losses, self.fitness_shaping)
        diff = utility[0::2] - utility[1::2]
        gradients = []
        for mod in self.modules:
            factors = noise[id(mod)]
            # Do not materialise perturbation matrices for individual candidates.
            # einsum directly accumulates the high/full rank update across pairs.
            grad = np.einsum("p,pir,por->io", diff, factors.a_pos, factors.b_pos, optimize=True)
            grad /= (2 * self.pairs * self.sigma * np.sqrt(self.rank))
            if self.weight_decay:
                grad += self.weight_decay * np.asarray(mod.weights)
            gradients.append(grad)
        grad_norm = float(np.sqrt(sum(np.sum(g*g) for g in gradients)))
        if self.grad_clip and grad_norm > self.grad_clip:
            gradients = [g * (self.grad_clip / grad_norm) for g in gradients]
        self.t += 1
        for i, (mod, grad) in enumerate(zip(self.modules, gradients)):
            if self.optimizer == "adam":
                self.m[i] = 0.9 * self.m[i] + 0.1 * grad
                self.v[i] = 0.999 * self.v[i] + 0.001 * (grad * grad)
                step = (self.m[i] / (1 - 0.9 ** self.t)) / (
                    np.sqrt(self.v[i] / (1 - 0.999 ** self.t)) + 1e-8
                )
            else:
                step = grad
            mod.weights = np.asarray(mod.weights, dtype=np.float64) - self.learning_rate * step
        return {"gradient_norm": grad_norm, "best_candidate_loss": float(losses.min()),
                "mean_candidate_loss": float(losses.mean())}


def _generic_population_step(layer, x, u, factors=None, sigma=0.):
    """Compatible vectorised step for original CMAES_FCLayer or EGGROLL_FCLayer."""
    if hasattr(layer, "population_step"):
        if factors is None:
            return layer.population_step(x, u)
        return layer.population_step(x, u, factors=(factors.a_full, factors.b_full), sigma=sigma)
    _check_compatible_layer(layer)
    current = x @ layer.weights
    if factors is not None:
        proj = np.einsum("pi,pir->pr", x, factors.a_full, optimize=True)
        current += sigma / np.sqrt(factors.a_full.shape[-1]) * np.einsum(
            "pr,por->po", proj, factors.b_full, optimize=True)
    u += current
    spikes = (u >= layer.threshold).astype(np.int64)
    u[spikes != 0] = 0.
    np.maximum(u, 0., out=u)
    u *= layer.decay
    return spikes


def _stage_targets(net, spike_train, stage):
    if stage is None or stage.target_mode == "global_input":
        return spike_train
    # Frozen layers feeding this stage produce identical inputs for all candidates.
    net.reset_state()
    targets = []
    first_train = min(stage.train_indices)
    for spikes in spike_train:
        x = spikes
        for layer in net.layers[:first_train]:
            x = layer(x)
        targets.append(np.asarray(x).copy())
    return np.asarray(targets)


def _population_sample_loss(net, spikes, target_spikes, noise, sigma, population_size,
                            trace_alpha, count_weight, delay, stage=None, temp_readout=None):
    """Low-memory streaming loss: no T x population x output raster allocation."""
    active_layers = (net.layers if stage is None else net.layers[:stage.active_end + 1])
    if temp_readout is not None:
        active_layers = list(active_layers) + [temp_readout]
    states = [np.zeros((population_size, _layer_output_size(layer)), dtype=np.float64)
              for layer in active_layers]
    for layer in active_layers:
        _check_compatible_layer(layer)
    target_traces = spike_trace(target_spikes, trace_alpha)
    t_count = len(spikes)
    if abs(delay) >= t_count:
        raise ValueError(f"abs(delay) must be < sequence length {t_count}")
    target_count = float(np.sum(target_spikes[max(0, -delay):min(t_count, t_count-delay)]))
    pred_count = np.zeros(population_size, dtype=np.float64)
    trace_sse = np.zeros(population_size, dtype=np.float64)
    output_trace = np.zeros((population_size, target_spikes.shape[1]), dtype=np.float64)
    for t, spike_t in enumerate(spikes):
        x = np.broadcast_to(spike_t, (population_size, len(spike_t)))
        for layer, state in zip(active_layers, states):
            x = _generic_population_step(layer, x, state, noise.get(id(layer)), sigma)
        if x.shape[1] != target_spikes.shape[1]:
            raise ValueError(f"Stage output {x.shape[1]} differs from target {target_spikes.shape[1]}")
        output_trace = trace_alpha * output_trace + x
        target_t = t - delay
        if 0 <= target_t < t_count:
            trace_sse += np.mean((output_trace - target_traces[target_t]) ** 2, axis=1)
            pred_count += x.sum(axis=1)
    trace_loss = trace_sse / (t_count - abs(delay))
    count_loss = ((pred_count - target_count) / (target_count + 1e-12)) ** 2
    return trace_loss + count_weight * count_loss


def evaluate_population(net, samples, perturbations, sigma, population_size,
                        trace_alpha=0.9, count_weight=0.01, delay=0,
                        stage=None, temp_readout=None):
    losses = np.zeros(population_size, dtype=np.float64)
    for batch in samples:
        spikes = extract_spike_train(batch, net.layers[0].input_num)
        targets = _stage_targets(net, spikes, stage)
        losses += _population_sample_loss(net, spikes, targets, perturbations, sigma,
                                          population_size, trace_alpha, count_weight, delay,
                                          stage=stage, temp_readout=temp_readout)
    return losses / len(samples)
