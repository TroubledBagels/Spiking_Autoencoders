"""Progressive symmetric GPU-accelerated EGGROLL; original schedule preserved.

CUDA -> Apple MPS -> CPU auto selection. Earlier stages stay frozen. Temporary
readouts are trained on the selected device and discarded after each stage.
"""
from __future__ import annotations

from pathlib import Path
import importlib

from EGGROLL_torch import TorchSpikingModel, TorchEGGROLL, device_description
from EGGROLL_utils import (
    SymmetricStage, network_widths, build_symmetric_stages,
    describe_symmetric_schedule, stage_trainable_modules, run_symmetric_stage,
    get_module_parameters, set_module_parameters, get_parameters, set_parameters,
    collect_samples, get_sample, extract_spike_train, evaluate_base_loss,
    plot_stage_sample, evaluate_test_metrics, test_silent_baseline, export_weights,
    delay_description, reconstruction_loss, reconstruction_loss_components,
    reconstruction_counts, reconstruction_counts_tolerant, calculate_prf,
)


def _normalise_stage_generations(stages, generations_per_stage):
    if isinstance(generations_per_stage, int):
        values = [generations_per_stage] * len(stages)
    else:
        values = list(generations_per_stage)
        if len(values) != len(stages):
            raise ValueError(f"Expected {len(stages)} stage generation counts, got {len(values)}")
    if any(x < 1 for x in values):
        raise ValueError("Every stage must have at least one generation")
    return values


def _make_temp_readout(net, stage):
    if not stage.use_temp_readout:
        return None
    layer_module = importlib.import_module(type(net.layers[0]).__module__)
    cls = getattr(layer_module, "EGGROLL_FCLayer", None)
    if cls is None:
        cls = getattr(layer_module, "CMAES_FCLayer", None)
    if cls is None:
        raise ImportError("Add EGGROLL_FCLayer to network_model/Layer.py")
    return cls(stage.temp_input_size, stage.temp_output_size)


def evaluate_stage_candidate(net, stage, temp_readout, theta, samples,
                             trace_alpha=.9, count_weight=.01, delay=0):
    modules = stage_trainable_modules(net, stage, temp_readout)
    set_module_parameters(modules, theta)
    return evaluate_base_loss(net, samples, trace_alpha, count_weight, delay,
                              stage=stage, temp_readout=temp_readout)


def evaluate_stage_metrics(net, stage, temp_readout, loader,
                           trace_alpha=.9, count_weight=.01, delay=0):
    return evaluate_test_metrics(net, loader, trace_alpha, count_weight, delay,
                                 stage=stage, temp_readout=temp_readout)


def train_symmetric(
    net, train_loader, test_loader, generations_per_stage=100,
    sigma=.15, samples_per_generation=16, test_samples=16,
    trace_alpha=.9, count_weight=.05, plot_sample_index=0,
    plot_dir="SymmetricEGGROLLEpochPlots", plot_every=1, delay=0,
    population_size=32, rank=4, learning_rate=.02,
    fitness_shaping="centered_rank", optimizer="adam", seed=None,
    weight_decay=0., grad_clip=None, restore_best=True,
    device="auto", dtype="float32",
):
    """GPU EGGROLL matching SymmetricCMAES stage order and target semantics."""
    stages = build_symmetric_stages(net)
    generations_list = _normalise_stage_generations(stages, generations_per_stage)
    backend = TorchSpikingModel(net, device=device, dtype=dtype)
    print(f"Full network: {' -> '.join(map(str, network_widths(net)))}")
    print(f"Device: {device_description(backend.device)} | dtype={dtype}")
    print(f"Delay: {delay} ({delay_description(delay)})")
    print(f"EGGROLL population={population_size}, rank={rank}, sigma={sigma}, lr={learning_rate}")
    describe_symmetric_schedule(stages)
    fixed_plot_batch = get_sample(test_loader, plot_sample_index) if plot_every else None
    if plot_every:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

    results = {"schedule": [stage.structure for stage in stages], "stages": []}
    for stage_num, (stage, n_gens) in enumerate(zip(stages, generations_list), start=1):
        temp_readout = _make_temp_readout(net, stage)
        if temp_readout is not None:
            backend.register(temp_readout)
        modules = stage_trainable_modules(net, stage, temp_readout)
        print("\n" + "=" * 74)
        print(f"STAGE {stage_num}/{len(stages)}: {stage.name}")
        print(f"Optimizing real layers {stage.train_indices}; total parameters: {get_module_parameters(modules).size}")
        if temp_readout is not None:
            print(f"Temporary readout {stage.temp_input_size} -> {stage.temp_output_size} "
                  "(discarded after stage)")
        print("=" * 74)
        engine = TorchEGGROLL(
            backend, modules, sigma=sigma, learning_rate=learning_rate,
            population_size=population_size, rank=rank,
            seed=None if seed is None else seed + stage_num - 1,
            fitness_shaping=fitness_shaping, optimizer=optimizer,
            weight_decay=weight_decay, grad_clip=grad_clip,
        )
        fixed_test_set = backend.prepare(collect_samples(test_loader, test_samples),
                                         stage=stage, trace_alpha=trace_alpha)
        stage_history = {
            "stage": stage_num, "name": stage.name, "structure": list(stage.structure),
            "train_layers": list(stage.train_indices),
            "train_loss": [], "test_loss": [],
            "best_candidate_loss": [], "gradient_norm": [],
        }
        best_train = float("inf")
        best_weights = backend.snapshot(modules)
        for generation in range(n_gens):
            samples = backend.prepare(collect_samples(train_loader, samples_per_generation),
                                      stage=stage, trace_alpha=trace_alpha)
            noise = engine.ask()
            losses = backend.evaluate(samples, population_size=population_size,
                                      noise=noise, sigma=sigma, trace_alpha=trace_alpha,
                                      count_weight=count_weight, delay=delay,
                                      stage=stage, temp_readout=temp_readout)
            update = engine.tell(noise, losses)
            train_loss = backend.base_loss(samples, trace_alpha=trace_alpha,
                                           count_weight=count_weight, delay=delay,
                                           stage=stage, temp_readout=temp_readout)
            test_loss = backend.base_loss(fixed_test_set, trace_alpha=trace_alpha,
                                          count_weight=count_weight, delay=delay,
                                          stage=stage, temp_readout=temp_readout)
            if train_loss < best_train:
                best_train = train_loss
                best_weights = backend.snapshot(modules)
            stage_history["train_loss"].append(train_loss)
            stage_history["test_loss"].append(test_loss)
            stage_history["best_candidate_loss"].append(update["best_candidate_loss"])
            stage_history["gradient_norm"].append(update["gradient_norm"])
            if plot_every and ((generation + 1) % plot_every == 0 or generation == 0):
                backend.sync_to_numpy()
                if temp_readout is not None:
                    backend.sync_to_numpy([temp_readout])
                plot_stage_sample(net, stage, temp_readout, get_module_parameters(modules),
                                  fixed_plot_batch, generation + 1, stage_num, plot_dir,
                                  trace_alpha, count_weight, delay)
            print(f"Stage {stage_num} generation {generation+1:03d}/{n_gens:03d} | "
                  f"train={train_loss:.6f} | test={test_loss:.6f} | "
                  f"candidate={update['best_candidate_loss']:.6f} | "
                  f"grad_norm={update['gradient_norm']:.4f}")
        if restore_best:
            backend.restore(modules, best_weights)
        backend.sync_to_numpy()              # trained stage layers frozen next stage
        if temp_readout is not None:
            backend.sync_to_numpy([temp_readout])
        stage_metrics = evaluate_stage_metrics(net, stage, temp_readout,
                                               test_loader, trace_alpha, count_weight, delay)
        stage_history["best_fitness"] = best_train
        stage_history["metrics"] = stage_metrics
        results["stages"].append(stage_history)
        print(f"Stage {stage_num} retained test loss: {stage_metrics['loss']:.6f} "
              f"| exact F1: {stage_metrics['f1']:.4f} "
              f"| +/-1: {stage_metrics['f1_1']:.4f} "
              f"| +/-2: {stage_metrics['f1_2']:.4f}")
        if temp_readout is not None:
            backend.unregister(temp_readout)
        del temp_readout, engine
        net.reset_state()

    backend.sync_to_numpy()
    silent_metrics = test_silent_baseline(net, test_loader, trace_alpha, count_weight, delay)
    metrics = evaluate_test_metrics(net, test_loader, trace_alpha, count_weight, delay)
    print("\nSYMMETRIC PROGRESSIVE EGGROLL TRAINING COMPLETE")
    print(f"Silent baseline: {silent_metrics['loss']:.6f}")
    print(f"Full test loss: {metrics['loss']:.6f} | trace: {metrics['trace_loss']:.6f} "
          f"| count: {metrics['count_loss']:.6f}")
    print(f"Exact F1: {metrics['f1']:.4f} | +/-1 F1: {metrics['f1_1']:.4f} "
          f"| +/-2 F1: {metrics['f1_2']:.4f}")
    print(f"Output/input spike ratio: {metrics['spike_ratio']:.4f}")
    return results, metrics, silent_metrics


def train(*args, **kwargs):
    return train_symmetric(*args, **kwargs)


if __name__ == "__main__":
    import torch
    from network_model import Layer as L, Network as N
    import SHDDataset as SHD

    local_dir = Path("~/data/shd/SHD").expanduser()
    tr_ds = SHD.SHDDataset(local_dir, train=True)
    te_ds = SHD.SHDDataset(local_dir, train=False)
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=1, shuffle=True)
    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=1, shuffle=False)
    model = N.Network()
    model.add([L.EGGROLL_FCLayer(700, 70), L.EGGROLL_FCLayer(70, 35),
               L.EGGROLL_FCLayer(35, 70), L.EGGROLL_FCLayer(70, 700)])
    results, metrics, silent = train_symmetric(
        model, tr_dl, te_dl, generations_per_stage=250,
        sigma=.15, population_size=32, rank=4, learning_rate=.002,
        samples_per_generation=16, test_samples=16, plot_every=10,
        device="auto",
    )
    export_weights(model)
