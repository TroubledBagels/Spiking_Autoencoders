
from dataclasses import dataclass
import pathlib
import gc
import warnings
import SHDDataset as SHD

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchaudio
import cma

import network_model.Layer as L
import network_model.Network as N
import network_model.GenerateData as GD
import EDDataset as edd

warnings.filterwarnings("ignore")


# =============================================================================
# Parameter helpers
# =============================================================================

def get_layers(net):
    return net.layers


def _layer_params(layer):
    if hasattr(layer, "get_params"):
        return np.asarray(layer.get_params(), dtype=np.float64).ravel()
    if hasattr(layer, "weights"):
        return np.asarray(layer.weights, dtype=np.float64).ravel()
    raise TypeError(f"Layer {type(layer).__name__} does not expose get_params() or weights")


def _set_layer_params(layer, theta):
    theta = np.asarray(theta, dtype=np.float64)
    if hasattr(layer, "set_params"):
        layer.set_params(theta)
        return
    if hasattr(layer, "weights"):
        layer.weights = theta.reshape(np.asarray(layer.weights).shape)
        return
    raise TypeError(f"Layer {type(layer).__name__} does not expose set_params() or weights")


def get_module_parameters(modules):
    params = [_layer_params(module) for module in modules]
    if not params:
        raise ValueError("No trainable modules supplied")
    return np.concatenate(params).astype(np.float64)


def set_module_parameters(modules, theta):
    theta = np.asarray(theta, dtype=np.float64)
    offset = 0

    for module in modules:
        size = _layer_params(module).size
        _set_layer_params(module, theta[offset:offset + size])
        offset += size

    if offset != theta.size:
        raise ValueError(f"Expected {offset} parameters, got {theta.size}")


def get_parameters(net):
    return get_module_parameters(net.layers)


def set_parameters(net, theta):
    set_module_parameters(net.layers, theta)


# =============================================================================
# Data / network execution helpers
# =============================================================================

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


def _reset_layer(layer):
    if hasattr(layer, "reset_state"):
        layer.reset_state()
    elif hasattr(layer, "reset"):
        layer.reset()


def run_autoencoder(net, spike_train):
    net.reset_state()

    latent_record = []
    output_record = []
    latent_index = max((len(net.layers) // 2) - 1, 0)

    for spikes_t in spike_train:
        x = spikes_t

        for i, layer in enumerate(net.layers):
            x = layer(x)
            if i == latent_index:
                latent_t = np.asarray(x).copy()

        latent_record.append(latent_t)
        output_record.append(np.asarray(x).copy())

    return np.asarray(latent_record), np.asarray(output_record)


# =============================================================================
# Loss / alignment / metrics
# =============================================================================

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
    """Align output[t] with input[t-delay], cropping invalid boundaries."""
    input_values = np.asarray(input_values)
    output_values = np.asarray(output_values)

    if input_values.shape != output_values.shape:
        raise ValueError(f"Input shape {input_values.shape} does not match output shape {output_values.shape}")

    timesteps = input_values.shape[0]
    if abs(delay) >= timesteps:
        raise ValueError(f"abs(delay) must be smaller than sequence length ({timesteps}), got {delay}")

    if delay > 0:
        return input_values[:-delay], output_values[delay:]
    if delay < 0:
        future = -delay
        return input_values[future:], output_values[:-future]
    return input_values, output_values


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
    trace_loss, count_loss = reconstruction_loss_components(
        input_spikes, output_spikes, trace_alpha=trace_alpha, delay=delay
    )
    return trace_loss + count_weight * count_loss


def reconstruction_counts(input_spikes, output_spikes, delay=0):
    input_spikes, output_spikes = align_for_delay(input_spikes, output_spikes, delay)

    target = input_spikes == 1
    predicted = output_spikes == 1

    tp = np.sum(target & predicted)
    fp = np.sum(~target & predicted)
    fn = np.sum(target & ~predicted)
    return tp, fp, fn


def reconstruction_counts_tolerant(input_spikes, output_spikes, tolerance=1, delay=0):
    """One-to-one spike matching within +/- tolerance timesteps per neuron."""
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


# =============================================================================
# Dataset helpers / plotting
# =============================================================================

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


def _save_reconstruction_plot(target_spikes, predicted_spikes, title, file_path, delay=0):
    target_spikes, predicted_spikes = align_for_delay(target_spikes, predicted_spikes, delay)

    tp, fp, fn = reconstruction_counts(target_spikes, predicted_spikes, delay=0)
    tp_1, fp_1, fn_1 = reconstruction_counts_tolerant(target_spikes, predicted_spikes, tolerance=1, delay=0)
    tp_2, fp_2, fn_2 = reconstruction_counts_tolerant(target_spikes, predicted_spikes, tolerance=2, delay=0)

    _, _, f1 = calculate_prf(tp, fp, fn)
    _, _, f1_1 = calculate_prf(tp_1, fp_1, fn_1)
    _, _, f1_2 = calculate_prf(tp_2, fp_2, fn_2)

    input_count = np.sum(target_spikes)
    output_count = np.sum(predicted_spikes)
    spike_ratio = output_count / (input_count + 1e-12)

    file_path = pathlib.Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)
    plot_spike_raster(axes[0], target_spikes, f"Target spikes | count={int(input_count)}")
    plot_spike_raster(
        axes[1], predicted_spikes,
        f"Output spikes | count={int(output_count)} | ratio={spike_ratio:.3f}"
    )
    axes[1].set_xlabel("Aligned timestep")
    fig.suptitle(f"{title} | exact F1={f1:.4f} | +/-1 F1={f1_1:.4f} | +/-2 F1={f1_2:.4f}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(file_path, dpi=200)
    plt.close(fig)
    del fig, axes
    gc.collect()


# =============================================================================
# Symmetric progressive schedule
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


def _layer_input_size(layer):
    if not hasattr(layer, "input_num"):
        raise TypeError(
            f"Layer {type(layer).__name__} needs input_num for automatic symmetric scheduling"
        )
    return int(layer.input_num)


def _layer_output_size(layer):
    for name in ("output_num", "output_size", "out_features"):
        if hasattr(layer, name):
            return int(getattr(layer, name))

    # Fallback for weight matrices. CMAES_FCLayer is expected to expose an output
    # size directly, but this keeps the schedule builder usable with simple layers.
    if hasattr(layer, "weights"):
        weights = np.asarray(layer.weights)
        input_size = _layer_input_size(layer)
        if weights.ndim == 2:
            if weights.shape[0] == input_size:
                return int(weights.shape[1])
            if weights.shape[1] == input_size:
                return int(weights.shape[0])

    raise TypeError(
        f"Layer {type(layer).__name__} needs output_num/output_size/out_features "
        "or a 2D weights matrix"
    )


def network_widths(net):
    if not net.layers:
        raise ValueError("Network contains no layers")

    widths = [_layer_input_size(net.layers[0])]
    for i, layer in enumerate(net.layers):
        in_size = _layer_input_size(layer)
        out_size = _layer_output_size(layer)

        if in_size != widths[-1]:
            raise ValueError(
                f"Layer {i} expects {in_size} inputs, but previous stage outputs {widths[-1]}"
            )
        widths.append(out_size)

    return widths


def build_symmetric_stages(net):
    """
    Reproduce the ProgressiveSpikingAutoencoderTrainer symmetric schedule.

    For widths I -> M -> N -> C -> N -> M -> I, the stages are:
        I -> M -> I
        I -> M -> N -> M
        I -> M -> N -> C -> N
        I -> M -> N -> C -> N -> M -> N
        I -> M -> N -> C -> N -> M -> I

    Temporary reconstruction layers are used for the incomplete outer stages and
    discarded once that stage has been retained.
    """
    widths = network_widths(net)
    layer_count = len(net.layers)

    if layer_count % 2 != 0:
        raise ValueError(
            "Symmetric progressive training expects an even number of synaptic layers "
            "(same encoder and decoder depth)"
        )

    depth = layer_count // 2
    if depth == 0:
        raise ValueError("Network needs at least one encoder and one decoder layer")

    if widths != list(reversed(widths)):
        raise ValueError(
            "Network widths are not symmetric. Expected a palindromic autoencoder shape, "
            f"got {' -> '.join(map(str, widths))}"
        )

    stages = []

    # Grow the encoder inward. Each new encoder layer is paired with a temporary
    # readout back to the activation size immediately before that layer.
    for encoder_index in range(depth - 1):
        previous_size = widths[encoder_index]
        new_size = widths[encoder_index + 1]
        structure = widths[:encoder_index + 2] + [previous_size]

        stages.append(
            SymmetricStage(
                name=" -> ".join(str(v) for v in structure),
                structure=structure,
                active_end=encoder_index,
                train_indices=[encoder_index],
                target_mode="local_input",
                use_temp_readout=True,
                temp_input_size=new_size,
                temp_output_size=previous_size,
            )
        )

    # Train the two centre layers together as the first real symmetric pair.
    center_encoder = depth - 1
    center_decoder = depth
    center_target_mode = "global_input" if depth == 1 else "local_input"
    center_structure = widths[:center_decoder + 2]

    stages.append(
        SymmetricStage(
            name=" -> ".join(str(v) for v in center_structure),
            structure=center_structure,
            active_end=center_decoder,
            train_indices=[center_encoder, center_decoder],
            target_mode=center_target_mode,
            use_temp_readout=False,
        )
    )

    # Grow outward through the decoder. Until the final output layer, each newly
    # added decoder layer gets a temporary readout to its own input resolution.
    for decoder_index in range(1, depth - 1):
        actual_index = depth + decoder_index
        previous_size = widths[actual_index]
        new_size = widths[actual_index + 1]
        structure = widths[:actual_index + 2] + [previous_size]

        stages.append(
            SymmetricStage(
                name=" -> ".join(str(v) for v in structure),
                structure=structure,
                active_end=actual_index,
                train_indices=[actual_index],
                target_mode="local_input",
                use_temp_readout=True,
                temp_input_size=new_size,
                temp_output_size=previous_size,
            )
        )

    # Finally attach the true output layer and train it against the original input.
    if depth > 1:
        final_index = layer_count - 1
        stages.append(
            SymmetricStage(
                name=" -> ".join(str(v) for v in widths),
                structure=widths,
                active_end=final_index,
                train_indices=[final_index],
                target_mode="global_input",
                use_temp_readout=False,
            )
        )

    return stages


def describe_symmetric_schedule(stages):
    print("Symmetric progressive schedule:")
    for i, stage in enumerate(stages, start=1):
        train_text = ", ".join(str(index) for index in stage.train_indices)
        suffix = " + temporary readout" if stage.use_temp_readout else ""
        print(f"  Stage {i}: {stage.name} | optimize real layer(s) [{train_text}]{suffix}")


def _make_temp_readout(stage):
    if not stage.use_temp_readout:
        return None
    return L.CMAES_FCLayer(stage.temp_input_size, stage.temp_output_size)


def run_symmetric_stage(net, spike_train, stage, temp_readout=None):
    """Run only the retained prefix needed for one progressive stage."""
    net.reset_state()
    if temp_readout is not None:
        _reset_layer(temp_readout)

    train_start = min(stage.train_indices)
    targets = []
    outputs = []

    for spikes_t in spike_train:
        current = np.asarray(spikes_t, dtype=np.float64)

        if stage.target_mode == "global_input":
            target_t = current.copy()
        else:
            target_t = None

        for index in range(stage.active_end + 1):
            if index == train_start and stage.target_mode == "local_input":
                target_t = np.asarray(current).copy()

            current = net.layers[index](current)

        if temp_readout is not None:
            current = temp_readout(current)

        if target_t is None:
            raise RuntimeError(f"No target was produced for stage {stage.name}")

        targets.append(target_t)
        outputs.append(np.asarray(current).copy())

    return np.asarray(targets), np.asarray(outputs)


def stage_trainable_modules(net, stage, temp_readout=None):
    modules = [net.layers[index] for index in stage.train_indices]
    if temp_readout is not None:
        modules.append(temp_readout)
    return modules


def evaluate_stage_candidate(
    net,
    stage,
    temp_readout,
    theta,
    samples,
    trace_alpha=0.9,
    count_weight=0.01,
    delay=0,
):
    modules = stage_trainable_modules(net, stage, temp_readout)
    set_module_parameters(modules, theta)

    total_loss = 0.0
    input_num = _layer_input_size(net.layers[0])

    for batch in samples:
        spike_train = extract_spike_train(batch, input_num)
        targets, outputs = run_symmetric_stage(net, spike_train, stage, temp_readout)
        total_loss += reconstruction_loss(targets, outputs, trace_alpha, count_weight, delay)

    return total_loss / max(len(samples), 1)


def evaluate_stage_metrics(net, stage, temp_readout, loader, trace_alpha=0.9, count_weight=0.01, delay=0):
    totals = {
        "loss": 0.0,
        "tp": 0, "fp": 0, "fn": 0,
        "tp_1": 0, "fp_1": 0, "fn_1": 0,
        "tp_2": 0, "fp_2": 0, "fn_2": 0,
        "count": 0,
    }
    input_num = _layer_input_size(net.layers[0])

    for batch in loader:
        spike_train = extract_spike_train(batch, input_num)
        targets, outputs = run_symmetric_stage(net, spike_train, stage, temp_readout)

        totals["loss"] += reconstruction_loss(targets, outputs, trace_alpha, count_weight, delay)

        tp, fp, fn = reconstruction_counts(targets, outputs, delay)
        tp_1, fp_1, fn_1 = reconstruction_counts_tolerant(targets, outputs, tolerance=1, delay=delay)
        tp_2, fp_2, fn_2 = reconstruction_counts_tolerant(targets, outputs, tolerance=2, delay=delay)

        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["tp_1"] += tp_1
        totals["fp_1"] += fp_1
        totals["fn_1"] += fn_1
        totals["tp_2"] += tp_2
        totals["fp_2"] += fp_2
        totals["fn_2"] += fn_2
        totals["count"] += 1

    p, r, f1 = calculate_prf(totals["tp"], totals["fp"], totals["fn"])
    p1, r1, f1_1 = calculate_prf(totals["tp_1"], totals["fp_1"], totals["fn_1"])
    p2, r2, f1_2 = calculate_prf(totals["tp_2"], totals["fp_2"], totals["fn_2"])

    return {
        "loss": totals["loss"] / max(totals["count"], 1),
        "precision": p,
        "recall": r,
        "f1": f1,
        "precision_1": p1,
        "recall_1": r1,
        "f1_1": f1_1,
        "precision_2": p2,
        "recall_2": r2,
        "f1_2": f1_2,
    }


def plot_stage_sample(
    net,
    stage,
    temp_readout,
    theta,
    batch,
    generation,
    stage_number,
    plot_dir,
    trace_alpha=0.9,
    count_weight=0.01,
    delay=0,
):
    modules = stage_trainable_modules(net, stage, temp_readout)
    set_module_parameters(modules, theta)

    spike_train = extract_spike_train(batch, _layer_input_size(net.layers[0]))
    targets, outputs = run_symmetric_stage(net, spike_train, stage, temp_readout)
    loss = reconstruction_loss(targets, outputs, trace_alpha, count_weight, delay)

    stage_dir = pathlib.Path(plot_dir) / f"stage_{stage_number:02d}"
    file_path = stage_dir / f"generation_{generation:03d}.png"
    _save_reconstruction_plot(
        targets,
        outputs,
        title=(
            f"Stage {stage_number}: {stage.name} | generation {generation:03d} | "
            f"loss={loss:.6f}"
        ),
        file_path=file_path,
        delay=delay,
    )


# =============================================================================
# Final full-network evaluation
# =============================================================================

def silent_output_loss(spike_train, trace_alpha=0.9, count_weight=0.01, delay=0):
    silent_output = np.zeros_like(spike_train)
    return reconstruction_loss(spike_train, silent_output, trace_alpha, count_weight, delay)


def test_silent_baseline(net, test_loader, trace_alpha=0.9, count_weight=0.01, delay=0):
    total_loss = 0.0
    total_trace_loss = 0.0
    total_count_loss = 0.0
    count = 0
    input_num = _layer_input_size(net.layers[0])

    for batch in test_loader:
        spike_train = extract_spike_train(batch, input_num)
        silent_output = np.zeros_like(spike_train)
        trace_loss, count_loss = reconstruction_loss_components(spike_train, silent_output, trace_alpha, delay)

        total_trace_loss += trace_loss
        total_count_loss += count_loss
        total_loss += trace_loss + count_weight * count_loss
        count += 1

    return {
        "loss": total_loss / max(count, 1),
        "trace_loss": total_trace_loss / max(count, 1),
        "count_loss": total_count_loss / max(count, 1),
    }


def evaluate_test_metrics(net, test_loader, trace_alpha=0.9, count_weight=0.01, delay=0):
    total_loss = 0.0
    total_trace_loss = 0.0
    total_count_loss = 0.0

    total_tp = total_fp = total_fn = 0
    total_tp_1 = total_fp_1 = total_fn_1 = 0
    total_tp_2 = total_fp_2 = total_fn_2 = 0

    total_input_spikes = 0
    total_output_spikes = 0
    count = 0

    for batch in test_loader:
        spike_train = extract_spike_train(batch, _layer_input_size(net.layers[0]))
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

    return {
        "loss": total_loss / max(count, 1),
        "trace_loss": total_trace_loss / max(count, 1),
        "count_loss": total_count_loss / max(count, 1),
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
        "spike_ratio": total_output_spikes / (total_input_spikes + 1e-12),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


# =============================================================================
# CMA-ES symmetric progressive training
# =============================================================================

def _normalise_stage_generations(stages, generations_per_stage):
    if isinstance(generations_per_stage, int):
        return [generations_per_stage] * len(stages)

    values = list(generations_per_stage)
    if len(values) != len(stages):
        raise ValueError(
            f"generations_per_stage must be an int or a sequence of length {len(stages)}"
        )
    return values


def train_symmetric(
    net,
    train_loader,
    test_loader,
    generations_per_stage=100,
    sigma=0.15,
    samples_per_generation=16,
    test_samples=16,
    trace_alpha=0.9,
    count_weight=0.05,
    plot_sample_index=0,
    plot_dir="SymmetricEpochPlots",
    plot_every=1,
    delay=0,
):
    """
    Layer-wise symmetric CMA-ES training.

    Frozen layers are never included in a stage's CMA-ES parameter vector. The
    temporary readout used by incomplete stages is optimized jointly with the new
    real layer, then discarded. The trained real layer remains in the model and is
    frozen for the next stage.
    """
    stages = build_symmetric_stages(net)
    stage_generations = _normalise_stage_generations(stages, generations_per_stage)

    print(f"Full network: {' -> '.join(map(str, network_widths(net)))}")
    print(f"Delay: {delay} ({delay_description(delay)})")
    describe_symmetric_schedule(stages)

    fixed_plot_batch = get_sample(test_loader, plot_sample_index)
    plot_dir = pathlib.Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "schedule": [stage.structure for stage in stages],
        "stages": [],
    }

    for stage_number, (stage, generations) in enumerate(zip(stages, stage_generations), start=1):
        temp_readout = _make_temp_readout(stage)
        modules = stage_trainable_modules(net, stage, temp_readout)
        initial_theta = get_module_parameters(modules)

        print("\n" + "=" * 100)
        print(f"STAGE {stage_number}/{len(stages)}: {stage.name}")
        print(f"Optimizing real layer(s): {stage.train_indices}")
        print(f"Stage CMA-ES parameters: {initial_theta.size}")
        if temp_readout is not None:
            print(
                f"Temporary readout: {stage.temp_input_size} -> {stage.temp_output_size} "
                "(discarded after this stage)"
            )
        print("=" * 100)

        es = cma.CMAEvolutionStrategy(initial_theta, sigma, {"verbose": -9})
        fixed_test_set = collect_samples(test_loader, test_samples)

        stage_history = {
            "stage": stage_number,
            "name": stage.name,
            "structure": list(stage.structure),
            "train_layers": list(stage.train_indices),
            "train_loss": [],
            "test_loss": [],
        }

        for generation in range(generations):
            train_samples = collect_samples(train_loader, samples_per_generation)
            candidates = es.ask()
            losses = []

            for theta in candidates:
                loss = evaluate_stage_candidate(
                    net,
                    stage,
                    temp_readout,
                    theta,
                    train_samples,
                    trace_alpha=trace_alpha,
                    count_weight=count_weight,
                    delay=delay,
                )
                losses.append(loss)

            es.tell(candidates, losses)

            best_generation_loss = float(min(losses))
            best_theta = np.asarray(es.result.xbest, dtype=np.float64).copy()
            test_loss = evaluate_stage_candidate(
                net,
                stage,
                temp_readout,
                best_theta,
                fixed_test_set,
                trace_alpha=trace_alpha,
                count_weight=count_weight,
                delay=delay,
            )

            stage_history["train_loss"].append(best_generation_loss)
            stage_history["test_loss"].append(float(test_loss))

            if plot_every and ((generation + 1) % plot_every == 0 or generation == 0):
                plot_stage_sample(
                    net,
                    stage,
                    temp_readout,
                    best_theta,
                    fixed_plot_batch,
                    generation + 1,
                    stage_number,
                    plot_dir,
                    trace_alpha=trace_alpha,
                    count_weight=count_weight,
                    delay=delay,
                )

            print(
                f"Stage {stage_number} generation {generation + 1:03d}/{generations:03d} | "
                f"train={best_generation_loss:.6f} | test={test_loss:.6f} | sigma={es.sigma:.6f}"
            )

            if es.stop():
                print(f"Stage {stage_number} stopping: {es.stop()}")
                break

        # Retain only the real network layers. If there was a temporary readout,
        # its part of xbest is simply ignored after this assignment.
        best_theta = np.asarray(es.result.xbest, dtype=np.float64).copy()
        set_module_parameters(modules, best_theta)

        real_modules = [net.layers[index] for index in stage.train_indices]
        retained_real_theta = get_module_parameters(real_modules)
        set_module_parameters(real_modules, retained_real_theta)

        # Evaluate the retained stage before discarding its temporary readout.
        stage_metrics = evaluate_stage_metrics(
            net,
            stage,
            temp_readout,
            test_loader,
            trace_alpha=trace_alpha,
            count_weight=count_weight,
            delay=delay,
        )

        stage_history["best_fitness"] = float(es.result.fbest)
        stage_history["metrics"] = stage_metrics
        results["stages"].append(stage_history)

        print(f"Stage {stage_number} retained test loss: {stage_metrics['loss']:.6f}")
        print(f"Stage {stage_number} exact F1: {stage_metrics['f1']:.4f}")
        print(f"Stage {stage_number} +/-1 F1: {stage_metrics['f1_1']:.4f}")
        print(f"Stage {stage_number} +/-2 F1: {stage_metrics['f1_2']:.4f}")

        # temp_readout is intentionally allowed to fall out of scope here.
        net.reset_state()

    silent_metrics = test_silent_baseline(net, test_loader, trace_alpha, count_weight, delay)
    metrics = evaluate_test_metrics(net, test_loader, trace_alpha, count_weight, delay)

    print("\n" + "#" * 100)
    print("SYMMETRIC PROGRESSIVE TRAINING COMPLETE")
    print("#" * 100)
    print(f"Silent baseline loss: {silent_metrics['loss']:.6f}")
    print(f"Full test loss:       {metrics['loss']:.6f}")
    print(f"Trace loss:           {metrics['trace_loss']:.6f}")
    print(f"Count loss:           {metrics['count_loss']:.6f}")
    print()
    print(f"Exact precision: {metrics['precision']:.4f}")
    print(f"Exact recall:    {metrics['recall']:.4f}")
    print(f"Exact F1:        {metrics['f1']:.4f}")
    print()
    print(f"+/-1 precision: {metrics['precision_1']:.4f}")
    print(f"+/-1 recall:    {metrics['recall_1']:.4f}")
    print(f"+/-1 F1:        {metrics['f1_1']:.4f}")
    print()
    print(f"+/-2 precision: {metrics['precision_2']:.4f}")
    print(f"+/-2 recall:    {metrics['recall_2']:.4f}")
    print(f"+/-2 F1:        {metrics['f1_2']:.4f}")
    print()
    print(f"Input spikes:  {metrics['input_spikes']}")
    print(f"Output spikes: {metrics['output_spikes']}")
    print(f"Output/input spike ratio: {metrics['spike_ratio']:.4f}")

    return results, metrics, silent_metrics


# Backward-friendly name if existing calling code expects train(...).
def train(*args, **kwargs):
    return train_symmetric(*args, **kwargs)


# =============================================================================
# Export
# =============================================================================

def export_weights(net, output_dir="ExportedWeights"):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, layer in enumerate(net.layers):
        if hasattr(layer, "weights"):
            weights = np.asarray(layer.weights)
            file_path = output_dir / f"layer_{i + 1}_weights.npy"
            np.save(file_path, weights)
            print(f"Saved layer {i + 1} weights to {file_path}")


# =============================================================================
# Example
# =============================================================================

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

    # Alternative synthetic dataset:
    # tr_ds = GD.SinWaveDS(20, 1000, 100, 500)
    # te_ds = GD.SinWaveDS(20, 1000, 100, 100)

    tr_ds = SHD.SHDDataset(local_dir, train=True)
    te_ds = SHD.SHDDataset(local_dir, train=False)

    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=1, shuffle=True)
    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=1, shuffle=False)

    model = N.Network()
    model.add(
        [
            L.CMAES_FCLayer(700, 35),
            # L.CMAES_FCLayer(16, 12),
            # L.CMAES_FCLayer(12, 16),
            L.CMAES_FCLayer(35, 700),
        ]
    )

    # For this 20 -> 16 -> 12 -> 16 -> 20 network the schedule is:
    #   Stage 1: 20 -> 16 -> 20        train layer 0 + temporary 16->20 readout
    #   Stage 2: 20 -> 16 -> 12 -> 16  train layers 1 and 2 together
    #   Stage 3: 20 -> 16 -> 12 -> 16 -> 20  train layer 3
    results, mets, silent_mets = train_symmetric(
        model,
        tr_dl,
        te_dl,
        generations_per_stage=250,
        sigma=0.15,
        samples_per_generation=16,
        test_samples=16,
        plot_sample_index=0,
        plot_dir="SymmetricEpochPlots",
        plot_every=1,
        delay=DELAY,
    )

    export_weights(model)
