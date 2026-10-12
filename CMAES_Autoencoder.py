import network_model.Layer as L
import network_model.Network as N
import EDDataset as edd
import pathlib
import numpy as np
import torch
import torchvision.transforms as transforms
import torchaudio
import cma
import warnings
import network_model.GenerateData as GD
import matplotlib
import SHDDataset as SHD

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import gc

warnings.filterwarnings("ignore")


def get_layers(net):
    return net.layers


def get_parameters(net):
    params = [layer.weights.ravel() for layer in get_layers(net) if hasattr(layer, "weights")]

    if not params:
        raise ValueError("No trainable weight matrices found")

    return np.concatenate(params).astype(np.float64)


def set_parameters(net, theta):
    theta = np.asarray(theta, dtype=np.float64)
    offset = 0

    for layer in net.layers:
        n = layer.get_params().size
        layer.set_params(theta[offset:offset + n])
        offset += n

    if offset != theta.size:
        raise ValueError(f"Expected {offset} parameters, got {theta.size}")


def extract_spike_train(batch, input_num):
    spikes = batch[0] if isinstance(batch, (tuple, list)) else batch

    if isinstance(spikes, torch.Tensor):
        spikes = spikes.detach().cpu().numpy()

    spikes = np.asarray(spikes)

    while spikes.ndim > 2 and spikes.shape[0] == 1:
        spikes = spikes[0]

    if spikes.ndim != 2:
        raise ValueError(f"Expected 2D spike train, got {spikes.shape}")

    if spikes.shape[1] == input_num:
        return spikes.astype(np.float64)

    if spikes.shape[0] == input_num:
        return spikes.T.astype(np.float64)

    raise ValueError(f"Spike train shape {spikes.shape} does not match {input_num} network inputs")


def run_autoencoder(net, spike_train):
    net.reset_state()

    latent_record = []
    output_record = []

    for spikes_t in spike_train:
        x = spikes_t

        for i, layer in enumerate(net.layers):
            x = layer(x)

            if i == 0:
                latent_t = x.copy()

        latent_record.append(latent_t)
        output_record.append(x.copy())

    return np.asarray(latent_record), np.asarray(output_record)


def spike_trace(spikes, alpha=0.9):
    spikes = np.asarray(spikes, dtype=np.float64)
    traces = np.zeros_like(spikes, dtype=np.float64)
    state = np.zeros(spikes.shape[1], dtype=np.float64)

    for t in range(spikes.shape[0]):
        state = alpha * state + spikes[t]
        traces[t] = state

    return traces


def delay_description(delay):
    if delay > 0:
        return f"reconstruct {delay} timestep(s) into the past"
    if delay < 0:
        return f"predict {-delay} timestep(s) into the future"
    return "reconstruct the current timestep"


def align_for_delay(input_values, output_values, delay=0):
    """
    Align network outputs with the desired input target.

    Sign convention:
        delay > 0:
            output[t] is trained against input[t - delay]
            -> reconstruct DELAY timesteps into the past.

        delay < 0:
            output[t] is trained against input[t + abs(delay)]
            -> predict abs(DELAY) timesteps into the future.

        delay == 0:
            output[t] is trained against input[t].

    Invalid boundary timesteps are cropped rather than padded, so they do not
    contribute to the loss or metrics.
    """
    input_values = np.asarray(input_values)
    output_values = np.asarray(output_values)

    if input_values.shape != output_values.shape:
        raise ValueError(f"Input shape {input_values.shape} does not match output shape {output_values.shape}")

    if input_values.ndim < 1:
        raise ValueError("Expected time to be the first dimension")

    timesteps = input_values.shape[0]

    if abs(delay) >= timesteps:
        raise ValueError(f"abs(delay) must be smaller than the sequence length ({timesteps}), got {delay}")

    if delay > 0:
        target = input_values[:-delay]
        predicted = output_values[delay:]
    elif delay < 0:
        future = -delay
        target = input_values[future:]
        predicted = output_values[:-future]
    else:
        target = input_values
        predicted = output_values

    return target, predicted


def reconstruction_loss_components(input_spikes, output_spikes, trace_alpha=0.9, delay=0):
    input_trace = spike_trace(input_spikes, trace_alpha)
    output_trace = spike_trace(output_spikes, trace_alpha)

    target_trace, predicted_trace = align_for_delay(input_trace, output_trace, delay)
    target_spikes, predicted_spikes = align_for_delay(input_spikes, output_spikes, delay)

    trace_loss = np.mean((target_trace - predicted_trace) ** 2)

    input_count = np.sum(target_spikes)
    output_count = np.sum(predicted_spikes)

    count_loss = ((output_count - input_count) / (input_count + 1e-12)) ** 2

    return float(trace_loss), float(count_loss)


def reconstruction_loss(input_spikes, output_spikes, trace_alpha=0.9, count_weight=0.01, delay=0):
    trace_loss, count_loss = reconstruction_loss_components(input_spikes, output_spikes, trace_alpha, delay)

    return trace_loss + count_weight * count_loss


def evaluate_candidate(net, theta, samples, trace_alpha=0.9, count_weight=0.01, delay=0):
    set_parameters(net, theta)

    total_loss = 0.0
    input_num = net.layers[0].input_num

    for batch in samples:
        spike_train = extract_spike_train(batch, input_num)
        _, output_spikes = run_autoencoder(net, spike_train)
        total_loss += reconstruction_loss(spike_train, output_spikes, trace_alpha, count_weight, delay)

    return total_loss / len(samples)


def collect_samples(loader, count):
    samples = []

    for batch in loader:
        samples.append(batch)

        if len(samples) >= count:
            break

    return samples


def get_sample(loader, sample_index=0):
    for index, batch in enumerate(loader):
        if index == sample_index:
            return batch

    raise IndexError(f"Sample index {sample_index} is outside the dataset")


def plot_spike_raster(ax, spikes, title):
    spikes = np.asarray(spikes)

    for neuron in range(spikes.shape[1]):
        spike_times = np.flatnonzero(spikes[:, neuron] == 1)

        if spike_times.size > 0:
            ax.scatter(spike_times, np.full(spike_times.shape, neuron), marker="|", s=18)

    ax.set_title(title)
    ax.set_ylabel("Neuron")
    ax.set_ylim(-0.5, spikes.shape[1] - 0.5)
    ax.set_yticks(np.arange(spikes.shape[1]))
    ax.grid(axis="x", alpha=0.2)


def plot_selected_sample(net, theta, spike_train, generation, plot_dir, trace_alpha=0.9, count_weight=0.01, delay=0):
    set_parameters(net, theta)
    _, output_spikes = run_autoencoder(net, spike_train)

    loss = reconstruction_loss(spike_train, output_spikes, trace_alpha, count_weight, delay)
    tp, fp, fn = reconstruction_counts(spike_train, output_spikes, delay)
    tp_1, fp_1, fn_1 = reconstruction_counts_tolerant(spike_train, output_spikes, tolerance=1, delay=delay)
    tp_2, fp_2, fn_2 = reconstruction_counts_tolerant(spike_train, output_spikes, tolerance=2, delay=delay)

    _, _, f1 = calculate_prf(tp, fp, fn)
    _, _, f1_1 = calculate_prf(tp_1, fp_1, fn_1)
    _, _, f1_2 = calculate_prf(tp_2, fp_2, fn_2)

    target_spikes, predicted_spikes = align_for_delay(spike_train, output_spikes, delay)
    input_count = np.sum(target_spikes)
    output_count = np.sum(predicted_spikes)
    spike_ratio = output_count / (input_count + 1e-12)

    plot_dir = pathlib.Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)

    plot_spike_raster(axes[0], target_spikes, f"Aligned target spikes | count={int(input_count)}")
    plot_spike_raster(axes[1], predicted_spikes,
                      f"Aligned output spikes | count={int(output_count)} | ratio={spike_ratio:.3f}")

    axes[1].set_xlabel("Aligned timestep")

    fig.suptitle(
        f"Generation {generation:03d} | delay={delay} ({delay_description(delay)}) | loss={loss:.6f} | "
        f"exact F1={f1:.4f} | +/-1 F1={f1_1:.4f} | +/-2 F1={f1_2:.4f}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_dir / f"generation_{generation:03d}.png", dpi=200)
    # plt.show()
    plt.close(fig)

    del fig
    del axes
    gc.collect()

    return output_spikes


def silent_output_loss(spike_train, trace_alpha=0.9, count_weight=0.01, delay=0):
    silent_output = np.zeros_like(spike_train)
    return reconstruction_loss(spike_train, silent_output, trace_alpha, count_weight, delay)


def reconstruction_counts(input_spikes, output_spikes, delay=0):
    input_spikes, output_spikes = align_for_delay(input_spikes, output_spikes, delay)

    target = input_spikes == 1
    predicted = output_spikes == 1

    tp = np.sum(target & predicted)
    fp = np.sum(~target & predicted)
    fn = np.sum(target & ~predicted)

    return tp, fp, fn


def reconstruction_counts_tolerant(input_spikes, output_spikes, tolerance=1, delay=0):
    input_spikes, output_spikes = align_for_delay(input_spikes, output_spikes, delay)

    total_tp = 0
    total_target = 0
    total_predicted = 0

    for neuron in range(input_spikes.shape[1]):
        target_times = np.flatnonzero(input_spikes[:, neuron] == 1)
        predicted_times = np.flatnonzero(output_spikes[:, neuron] == 1)

        total_target += len(target_times)
        total_predicted += len(predicted_times)

        target_index = 0
        predicted_index = 0

        while target_index < len(target_times) and predicted_index < len(predicted_times):
            target_time = target_times[target_index]
            predicted_time = predicted_times[predicted_index]

            if abs(target_time - predicted_time) <= tolerance:
                total_tp += 1
                target_index += 1
                predicted_index += 1
            elif predicted_time < target_time - tolerance:
                predicted_index += 1
            else:
                target_index += 1

    total_fp = total_predicted - total_tp
    total_fn = total_target - total_tp

    return total_tp, total_fp, total_fn


def calculate_prf(tp, fp, fn):
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return precision, recall, f1


def evaluate_test_metrics(net, test_loader, trace_alpha=0.9, count_weight=0.01, delay=0):
    total_loss = 0.0
    total_trace_loss = 0.0
    total_count_loss = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_tp_1 = 0
    total_fp_1 = 0
    total_fn_1 = 0

    total_tp_2 = 0
    total_fp_2 = 0
    total_fn_2 = 0

    total_input_spikes = 0
    total_output_spikes = 0
    count = 0

    for batch in test_loader:
        spike_train = extract_spike_train(batch, net.layers[0].input_num)
        _, output_spikes = run_autoencoder(net, spike_train)

        trace_loss, count_loss = reconstruction_loss_components(spike_train, output_spikes, trace_alpha, delay)
        loss = trace_loss + count_weight * count_loss

        tp, fp, fn = reconstruction_counts(spike_train, output_spikes, delay)
        tp_1, fp_1, fn_1 = reconstruction_counts_tolerant(spike_train, output_spikes, tolerance=1, delay=delay)
        tp_2, fp_2, fn_2 = reconstruction_counts_tolerant(spike_train, output_spikes, tolerance=2, delay=delay)

        total_loss += loss
        total_trace_loss += trace_loss
        total_count_loss += count_loss

        total_tp += tp
        total_fp += fp
        total_fn += fn

        total_tp_1 += tp_1
        total_fp_1 += fp_1
        total_fn_1 += fn_1

        total_tp_2 += tp_2
        total_fp_2 += fp_2
        total_fn_2 += fn_2

        target_spikes, predicted_spikes = align_for_delay(spike_train, output_spikes, delay)
        total_input_spikes += np.sum(target_spikes)
        total_output_spikes += np.sum(predicted_spikes)

        count += 1

    precision, recall, f1 = calculate_prf(total_tp, total_fp, total_fn)
    precision_1, recall_1, f1_1 = calculate_prf(total_tp_1, total_fp_1, total_fn_1)
    precision_2, recall_2, f1_2 = calculate_prf(total_tp_2, total_fp_2, total_fn_2)

    spike_ratio = total_output_spikes / (total_input_spikes + 1e-12)

    return {
        "loss": total_loss / count,
        "trace_loss": total_trace_loss / count,
        "count_loss": total_count_loss / count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
        "precision_2": precision_2,
        "recall_2": recall_2,
        "f1_2": f1_2,
        "input_spikes": total_input_spikes,
        "output_spikes": total_output_spikes,
        "spike_ratio": spike_ratio,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }


def test(net, test_loader, sample_count=16, trace_alpha=0.9, count_weight=0.01, delay=0):
    samples = collect_samples(test_loader, sample_count)
    theta = get_parameters(net)

    return evaluate_candidate(net, theta, samples, trace_alpha, count_weight, delay)


def test_silent_baseline(net, test_loader, trace_alpha=0.9, count_weight=0.01, delay=0):
    total_loss = 0.0
    total_trace_loss = 0.0
    total_count_loss = 0.0
    count = 0
    input_num = net.layers[0].input_num

    for batch in test_loader:
        spike_train = extract_spike_train(batch, input_num)
        silent_output = np.zeros_like(spike_train)

        trace_loss, count_loss = reconstruction_loss_components(spike_train, silent_output, trace_alpha, delay)

        total_trace_loss += trace_loss
        total_count_loss += count_loss
        total_loss += trace_loss + count_weight * count_loss

        count += 1

    return {
        "loss": total_loss / count,
        "trace_loss": total_trace_loss / count,
        "count_loss": total_count_loss / count
    }


def train(net, train_loader, test_loader, generations=100,
          sigma=0.15, samples_per_generation=16, test_samples=16,
          trace_alpha=0.9, count_weight=0.05, plot_sample_index=0,
          plot_dir="EpochPlots", delay=0):
    initial_theta = get_parameters(net)

    print(f"Training {len(net.layers)} layers")
    print(f"CMA-ES parameters: {initial_theta.size}")
    print(f"Delay: {delay} ({delay_description(delay)})")

    es = cma.CMAEvolutionStrategy(
        initial_theta,
        sigma,
        {
            "verbose": -9,
            "CMA_diagonal": True,
            "popsize": 8
        }
    )
    fixed_test_set = collect_samples(test_loader, test_samples)

    fixed_plot_batch = get_sample(test_loader, plot_sample_index)
    fixed_plot_sample = extract_spike_train(fixed_plot_batch, net.layers[0].input_num)
    plot_dir = pathlib.Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting fixed test sample index: {plot_sample_index}")
    print(f"Epoch plots: {plot_dir.resolve()}")

    history = {"train_loss": [], "test_loss": []}

    for generation in range(generations):
        train_samples = collect_samples(train_loader, samples_per_generation)
        candidates = es.ask()
        losses = []

        for theta in candidates:
            loss = evaluate_candidate(net, theta, train_samples, trace_alpha, count_weight, delay)
            losses.append(loss)

        es.tell(candidates, losses)

        best_generation_loss = min(losses)
        best_theta = np.asarray(es.result.xbest).copy()
        test_loss = evaluate_candidate(net, best_theta, fixed_test_set, trace_alpha, count_weight, delay)

        history["train_loss"].append(best_generation_loss)
        history["test_loss"].append(test_loss)

        plot_selected_sample(net, best_theta, fixed_plot_sample, generation + 1, plot_dir, trace_alpha, count_weight,
                             delay)

        print(
            f"Generation {generation + 1:03d} | train={best_generation_loss:.6f} | test={test_loss:.6f} | sigma={es.sigma:.6f}")

        if es.stop():
            print(f"Stopping: {es.stop()}")
            break

    set_parameters(net, es.result.xbest)
    net.reset_state()

    silent_metrics = test_silent_baseline(net, test_loader, trace_alpha, count_weight, delay)
    metrics = evaluate_test_metrics(net, test_loader, trace_alpha, count_weight, delay)

    print()
    print(f"Silent baseline: {silent_metrics['loss']:.6f}")
    print(f"Silent trace loss: {silent_metrics['trace_loss']:.6f}")
    print(f"Silent count loss: {silent_metrics['count_loss']:.6f}")
    print()
    print(f"Best fitness: {es.result.fbest:.6f}")
    print(f"Full test loss: {metrics['loss']:.6f}")
    print(f"Trace loss: {metrics['trace_loss']:.6f}")
    print(f"Count loss: {metrics['count_loss']:.6f}")
    print()
    print(f"Exact precision: {metrics['precision']:.4f}")
    print(f"Exact recall: {metrics['recall']:.4f}")
    print(f"Exact F1: {metrics['f1']:.4f}")
    print()
    print(f"+/-1 precision: {metrics['precision_1']:.4f}")
    print(f"+/-1 recall: {metrics['recall_1']:.4f}")
    print(f"+/-1 F1: {metrics['f1_1']:.4f}")
    print()
    print(f"+/-2 precision: {metrics['precision_2']:.4f}")
    print(f"+/-2 recall: {metrics['recall_2']:.4f}")
    print(f"+/-2 F1: {metrics['f1_2']:.4f}")
    print()
    print(f"Input spikes: {metrics['input_spikes']}")
    print(f"Output spikes: {metrics['output_spikes']}")
    print(f"Output/input spike ratio: {metrics['spike_ratio']:.4f}")

    return history, metrics, silent_metrics


def export_weights(net, output_dir="ExportedWeights"):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, layer in enumerate(net.layers):
        if hasattr(layer, "weights"):
            weights = np.asarray(layer.weights)
            file_path = output_dir / f"layer_{i + 1}_weights.npy"

            np.save(file_path, weights)

            print(f"Saved layer {i + 1} weights to {file_path}")


if __name__ == "__main__":
    home = pathlib.Path("~").expanduser()
    # local_dir = home / "data" / "URBAN-SED"
    local_dir = home / "data" / "shd" / "SHD"

    DELAY = 0

    a_transform = transforms.Compose([
        torchaudio.transforms.Resample(44100, 16000),
        edd.UnsqueezeTransform(dim=0),
        edd.ToSpikeTransform(num_channels=10),
        edd.SqueezeTransform(dim=0),
        edd.SqueezeTransform(dim=0),
    ])

    # tr_ds = edd.URBANDataset(local_dir, split=edd.DatasetSplit.TRAIN, transform=a_transform)
    # te_ds = edd.URBANDataset(local_dir, split=edd.DatasetSplit.TEST, transform=a_transform)
    #
    # tr_ds = GD.SinWaveDS(20, 1000, 100, 500)
    # te_ds = GD.SinWaveDS(20, 1000, 100, 100)

    tr_ds = SHD.SHDDataset(local_dir, train=True)
    te_ds = SHD.SHDDataset(local_dir, train=False)

    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=1, shuffle=True)
    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=1, shuffle=False)

    model = N.Network()
    model.add(
        [
            L.CMAES_FCLayer(700, 70),
            # L.CMAES_FCLayer(16, 12),
            # L.CMAES_FCLayer(12, 16),
            L.CMAES_FCLayer(70, 700),
        ]
    )

    model.layers[0].weights = np.load("./ExportedWeights/layer_1_weights.npy")
    model.layers[1].weights = np.load("./ExportedWeights/layer_2_weights.npy")

    loss_hist, mets, s_mets = train(model, tr_dl, te_dl, plot_sample_index=0, plot_dir="EpochPlots", generations=250,
                                    delay=DELAY)

    export_weights(model)