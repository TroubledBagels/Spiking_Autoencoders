"""Whole-network GPU-accelerated EGGROLL; original train(...) API preserved.

Automatically selects NVIDIA CUDA -> Apple Silicon MPS -> PyTorch CPU.
Population dynamics, losses and optimizer run on selected device. The original
NumPy layers are updated for plotting, final testing and export.
"""
from __future__ import annotations

from pathlib import Path

from EGGROLL_torch import TorchSpikingModel, TorchEGGROLL, device_description
from EGGROLL_utils import (
    collect_samples, get_sample, extract_spike_train, get_parameters,
    set_parameters, evaluate_candidate, evaluate_base_loss, run_autoencoder,
    plot_selected_sample, test_silent_baseline, evaluate_test_metrics,
    export_weights, reconstruction_loss, reconstruction_loss_components,
    delay_description, reconstruction_counts, reconstruction_counts_tolerant,
    calculate_prf, spike_trace, align_for_delay,
)


def test(net, test_loader, sample_count=16, trace_alpha=.9, count_weight=.01, delay=0):
    samples = collect_samples(test_loader, sample_count)
    return evaluate_base_loss(net, samples, trace_alpha, count_weight, delay)


def train(
    net, train_loader, test_loader, generations=100,
    sigma=.15, samples_per_generation=16, test_samples=16,
    trace_alpha=.9, count_weight=.05, plot_sample_index=0,
    plot_dir="EGGROLLEpochPlots", delay=0,
    population_size=32, rank=4, learning_rate=.02,
    fitness_shaping="centered_rank", optimizer="adam", seed=None,
    weight_decay=0., grad_clip=None, plot_every=1, restore_best=True,
    device="auto", dtype="float32",
):
    """Optimize all spiking FC layers jointly using GPU/MPS/CPU EGGROLL.

    Same returns and argument names as the original trainer; new optional args:
    device='auto'|'cuda'|'mps'|'cpu' and dtype='float32'|'float64'.
    Layers may be either CMAES_FCLayer or EGGROLL_FCLayer.
    """
    print("Training EGGROLL...")
    if generations < 1:
        raise ValueError("generations must be >= 1")
    backend = TorchSpikingModel(net, device=device, dtype=dtype)
    print("Intialised model to device")
    engine = TorchEGGROLL(
        backend, net.layers, sigma=sigma, learning_rate=learning_rate,
        population_size=population_size, rank=rank, seed=seed,
        fitness_shaping=fitness_shaping, optimizer=optimizer,
        weight_decay=weight_decay, grad_clip=grad_clip,
    )
    print(f"Training {len(net.layers)} layers with whole-network EGGROLL")
    print(f"Device: {device_description(backend.device)} | dtype={dtype}")
    print(f"EGGROLL parameters: {get_parameters(net).size}")
    print(f"Population: {population_size} | rank: {rank} | sigma: {sigma} | lr: {learning_rate}")
    print(f"Delay: {delay} ({delay_description(delay)})")

    fixed_test_set = backend.prepare(collect_samples(test_loader, test_samples), trace_alpha=trace_alpha)
    fixed_plot_sample = None
    if plot_every:
        fixed_plot_sample = extract_spike_train(get_sample(test_loader, plot_sample_index),
                                                 net.layers[0].input_num)
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        print(f"Plotting fixed test sample index: {plot_sample_index}")
        print(f"Epoch plots: {Path(plot_dir).resolve()}")

    history = {"train_loss": [], "test_loss": [], "best_candidate_loss": [], "gradient_norm": []}
    best_train_loss = float("inf")
    best_weights = backend.snapshot(net.layers)

    for generation in range(generations):
        samples = backend.prepare(collect_samples(train_loader, samples_per_generation),
                                  trace_alpha=trace_alpha)
        noise = engine.ask()
        losses = backend.evaluate(samples, population_size=population_size, noise=noise,
                                  sigma=sigma, trace_alpha=trace_alpha,
                                  count_weight=count_weight, delay=delay)
        update = engine.tell(noise, losses)
        # Unperturbed base model is evaluated on GPU; no CPU forward passes.
        train_loss = backend.base_loss(samples, trace_alpha=trace_alpha,
                                       count_weight=count_weight, delay=delay)
        test_loss = backend.base_loss(fixed_test_set, trace_alpha=trace_alpha,
                                      count_weight=count_weight, delay=delay)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_weights = backend.snapshot(net.layers)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["best_candidate_loss"].append(update["best_candidate_loss"])
        history["gradient_norm"].append(update["gradient_norm"])
        if plot_every and ((generation + 1) % plot_every == 0 or generation == 0):
            backend.sync_to_numpy()
            plot_selected_sample(net, get_parameters(net), fixed_plot_sample,
                                 generation + 1, plot_dir, trace_alpha, count_weight, delay)
        print(f"Generation {generation+1:03d}/{generations:03d} | "
              f"train={train_loss:.6f} | test={test_loss:.6f} | "
              f"candidate={update['best_candidate_loss']:.6f} | "
              f"grad_norm={update['gradient_norm']:.4f}")

    if restore_best:
        backend.restore(net.layers, best_weights)
    backend.sync_to_numpy()
    net.reset_state()
    silent_metrics = test_silent_baseline(net, test_loader, trace_alpha, count_weight, delay)
    metrics = evaluate_test_metrics(net, test_loader, trace_alpha, count_weight, delay)
    print("\nWHOLE-NETWORK EGGROLL TRAINING COMPLETE")
    print(f"Best observed training loss: {best_train_loss:.6f}")
    print(f"Silent baseline: {silent_metrics['loss']:.6f}")
    print(f"Full test loss: {metrics['loss']:.6f} | trace: {metrics['trace_loss']:.6f} "
          f"| count: {metrics['count_loss']:.6f}")
    print(f"Exact F1: {metrics['f1']:.4f} | +/-1 F1: {metrics['f1_1']:.4f} "
          f"| +/-2 F1: {metrics['f1_2']:.4f}")
    print(f"Output/input spike ratio: {metrics['spike_ratio']:.4f}")
    return history, metrics, silent_metrics


if __name__ == "__main__":
    print("Starting EGGROLL Autoencoder")
    import torch
    print("Imported torch")
    from network_model import Layer as L, Network as N
    print("Imported network")
    import SHDDataset as SHD
    print("Imported dataset")

    local_dir = Path("~/data/shd/SHD").expanduser()
    print("Local directory", local_dir)
    train_ds = SHD.SHDDataset(local_dir, train=True)
    print("Loaded train dataset")
    test_ds = SHD.SHDDataset(local_dir, train=False)
    print("Loaded test dataset")
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    print("Loaded train dataloader")
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
    print("Loaded test dataloader")
    model = N.Network()
    print("Created model")
    model.add([
        L.EGGROLL_FCLayer(700, 222),
        L.EGGROLL_FCLayer(222, 70),
        L.EGGROLL_FCLayer(70, 222),
        L.EGGROLL_FCLayer(222, 700),
    ])
    print("Added layers")
    history, metrics, silent = train(
        model, train_dl, test_dl, generations=1000,
        sigma=.03, population_size=128, rank=4, learning_rate=.002,
        samples_per_generation=32, test_samples=16, plot_every=10, delay=0,
        device="auto",
    )
    export_weights(model)
