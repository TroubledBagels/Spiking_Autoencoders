import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
import csv

from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

INPUT_SIZE = 10
LATENT_SIZE = 5

VALIDATION_FRACTION = 0.2
SPLIT_SEED = 12345

MAX_GENERATIONS = 250
EARLY_STOPPING_PATIENCE = 40
VALIDATION_SAMPLES = 16

SEEDS = [101, 202, 303, 404, 505]

MAX_WORKERS = min(8, os.cpu_count() or 1)

SELECTION_METRIC = "f1_2"

HISTORY_FIELDS = [
    "configuration_id",
    "run_id",
    "stage",
    "seed",
    "generation",
    "train_loss",
    "validation_loss",
    "best_validation_loss",
    "sigma"
]

CONFIGURATION_FIELDS = [
    "configuration_id",
    "stage",
    "count_weight",
    "trace_alpha",
    "sigma",
    "samples_per_generation",
    "population_size",
    "decay",
    "threshold"
]

PARAMETER_KEYS = [
    "count_weight",
    "trace_alpha",
    "sigma",
    "samples_per_generation",
    "population_size",
    "decay",
    "threshold"
]

RESULT_FIELDS = [
    "configuration_id",
    "run_id",
    "history_file",
    "stage",
    "split",
    "seed",
    "count_weight",
    "trace_alpha",
    "sigma",
    "samples_per_generation",
    "population_size",
    "decay",
    "threshold",
    "max_generations",
    "generations_run",
    "best_training_fitness",
    "best_validation_loss",
    "loss",
    "trace_loss",
    "count_loss",
    "precision",
    "recall",
    "f1",
    "precision_1",
    "recall_1",
    "f1_1",
    "precision_2",
    "recall_2",
    "f1_2",
    "input_spikes",
    "output_spikes",
    "spike_ratio",
    "silent_loss",
    "silent_trace_loss",
    "silent_count_loss"
]


def encode_id_value(value):
    text = f"{value:g}" if isinstance(value, float) else str(value)

    return text.replace("-", "m").replace(".", "p")


def decode_id_value(value):
    return value.replace("m", "-").replace("p", ".")


def make_configuration_id(stage, params):
    return (
        f"{stage}"
        f"__cw{encode_id_value(params['count_weight'])}"
        f"__ta{encode_id_value(params['trace_alpha'])}"
        f"__sig{encode_id_value(params['sigma'])}"
        f"__spg{encode_id_value(params['samples_per_generation'])}"
        f"__pop{encode_id_value(params['population_size'])}"
        f"__dec{encode_id_value(params['decay'])}"
        f"__th{encode_id_value(params['threshold'])}"
    )


def make_run_id(configuration_id, seed):
    return f"{configuration_id}__seed{seed}"


def parse_configuration_id(configuration_id):
    parts = configuration_id.split("__")

    stage = parts[0]
    values = {}

    for part in parts[1:]:
        if part.startswith("cw"):
            values["count_weight"] = float(decode_id_value(part[2:]))
        elif part.startswith("ta"):
            values["trace_alpha"] = float(decode_id_value(part[2:]))
        elif part.startswith("sig"):
            values["sigma"] = float(decode_id_value(part[3:]))
        elif part.startswith("spg"):
            values["samples_per_generation"] = int(decode_id_value(part[3:]))
        elif part.startswith("pop"):
            values["population_size"] = int(decode_id_value(part[3:]))
        elif part.startswith("dec"):
            values["decay"] = float(decode_id_value(part[3:]))
        elif part.startswith("th"):
            values["threshold"] = float(decode_id_value(part[2:]))

    return stage, values


def parse_run_id(run_id):
    configuration_id, seed = run_id.rsplit("__seed", 1)
    stage, params = parse_configuration_id(configuration_id)

    return stage, params, int(seed)


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


def reconstruction_loss_components(input_spikes, output_spikes, trace_alpha=0.9):
    input_trace = spike_trace(input_spikes, trace_alpha)
    output_trace = spike_trace(output_spikes, trace_alpha)

    trace_loss = np.mean((input_trace - output_trace) ** 2)

    input_count = np.sum(input_spikes)
    output_count = np.sum(output_spikes)

    count_loss = ((output_count - input_count) / (input_count + 1e-12)) ** 2

    return float(trace_loss), float(count_loss)


def reconstruction_loss(input_spikes, output_spikes, trace_alpha=0.9, count_weight=0.01):
    trace_loss, count_loss = reconstruction_loss_components(input_spikes, output_spikes, trace_alpha)

    return trace_loss + count_weight * count_loss


def evaluate_candidate(net, theta, samples, trace_alpha=0.9, count_weight=0.01):
    set_parameters(net, theta)

    total_loss = 0.0
    input_num = net.layers[0].input_num

    for batch in samples:
        spike_train = extract_spike_train(batch, input_num)
        _, output_spikes = run_autoencoder(net, spike_train)
        total_loss += reconstruction_loss(spike_train, output_spikes, trace_alpha, count_weight)

    return total_loss / len(samples)


def collect_samples(loader, count):
    samples = []

    for batch in loader:
        samples.append(batch)

        if len(samples) >= count:
            break

    return samples


def reconstruction_counts(input_spikes, output_spikes):
    target = input_spikes == 1
    predicted = output_spikes == 1

    tp = np.sum(target & predicted)
    fp = np.sum(~target & predicted)
    fn = np.sum(target & ~predicted)

    return tp, fp, fn


def reconstruction_counts_tolerant(input_spikes, output_spikes, tolerance=1):
    input_spikes = np.asarray(input_spikes)
    output_spikes = np.asarray(output_spikes)

    if input_spikes.shape != output_spikes.shape:
        raise ValueError(f"Input shape {input_spikes.shape} does not match output shape {output_spikes.shape}")

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


def evaluate_test_metrics(net, data_loader, trace_alpha=0.9, count_weight=0.01):
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

    for batch in data_loader:
        spike_train = extract_spike_train(batch, net.layers[0].input_num)
        _, output_spikes = run_autoencoder(net, spike_train)

        trace_loss, count_loss = reconstruction_loss_components(spike_train, output_spikes, trace_alpha)
        loss = trace_loss + count_weight * count_loss

        tp, fp, fn = reconstruction_counts(spike_train, output_spikes)
        tp_1, fp_1, fn_1 = reconstruction_counts_tolerant(spike_train, output_spikes, tolerance=1)
        tp_2, fp_2, fn_2 = reconstruction_counts_tolerant(spike_train, output_spikes, tolerance=2)

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

        total_input_spikes += np.sum(spike_train)
        total_output_spikes += np.sum(output_spikes)

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
        "spike_ratio": spike_ratio
    }


def test_silent_baseline(net, data_loader, trace_alpha=0.9, count_weight=0.01):
    total_loss = 0.0
    total_trace_loss = 0.0
    total_count_loss = 0.0
    count = 0
    input_num = net.layers[0].input_num

    for batch in data_loader:
        spike_train = extract_spike_train(batch, input_num)
        silent_output = np.zeros_like(spike_train)

        trace_loss, count_loss = reconstruction_loss_components(spike_train, silent_output, trace_alpha)

        total_trace_loss += trace_loss
        total_count_loss += count_loss
        total_loss += trace_loss + count_weight * count_loss

        count += 1

    return {
        "loss": total_loss / count,
        "trace_loss": total_trace_loss / count,
        "count_loss": total_count_loss / count
    }


def build_model(params):
    model = N.Network()

    encoder = L.CMAES_FCLayer(INPUT_SIZE, LATENT_SIZE, decay=params["decay"])
    decoder = L.CMAES_FCLayer(LATENT_SIZE, INPUT_SIZE, decay=params["decay"])

    encoder.threshold = params["threshold"]
    decoder.threshold = params["threshold"]

    model.add([
        encoder,
        decoder
    ])

    return model


def make_transform():
    return transforms.Compose([
        torchaudio.transforms.Resample(44100, 16000),
        edd.UnsqueezeTransform(dim=0),
        edd.ToSpikeTransform(num_channels=5),
        edd.SqueezeTransform(dim=0),
        edd.SqueezeTransform(dim=0),
    ])


def make_loaders(local_dir, seed):
    a_transform = make_transform()

    full_train_ds = edd.URBANDataset(local_dir, split=edd.DatasetSplit.TRAIN, transform=a_transform)
    test_ds = edd.URBANDataset(local_dir, split=edd.DatasetSplit.TEST, transform=a_transform)

    validation_size = int(len(full_train_ds) * VALIDATION_FRACTION)
    train_size = len(full_train_ds) - validation_size

    split_generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_ds, validation_ds = torch.utils.data.random_split(full_train_ds, [train_size, validation_size], generator=split_generator)

    train_generator = torch.Generator().manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, generator=train_generator, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(validation_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    return train_loader, validation_loader, test_loader


def initialise_history_file(history_file):
    history_file.parent.mkdir(parents=True, exist_ok=True)

    with open(history_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()


def save_history_row(history_file, configuration_id, run_id, stage, seed, generation, train_loss, validation_loss, best_validation_loss, sigma):
    result = {
        "configuration_id": configuration_id,
        "run_id": run_id,
        "stage": stage,
        "seed": seed,
        "generation": generation,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "sigma": sigma
    }

    with open(history_file, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writerow(result)


def train(net, train_loader, validation_loader, generations=250, patience=40, sigma=0.05, samples_per_generation=8, population_size=17, validation_samples=16, trace_alpha=0.9, count_weight=0.01, seed=101, verbose=False, configuration_id=None, run_id=None, stage=None, history_file=None):
    initial_theta = get_parameters(net)

    es = cma.CMAEvolutionStrategy(initial_theta, sigma, {
        "popsize": population_size,
        "seed": seed,
        "verbose": -9
    })

    fixed_validation_set = collect_samples(validation_loader, validation_samples)

    best_validation_theta = initial_theta.copy()
    best_validation_loss = evaluate_candidate(net, initial_theta, fixed_validation_set, trace_alpha, count_weight)

    generations_without_improvement = 0
    generations_run = 0

    history = {
        "train_loss": [],
        "validation_loss": []
    }

    if history_file is not None:
        initialise_history_file(history_file)

    for generation in range(generations):
        train_samples = collect_samples(train_loader, samples_per_generation)
        candidates = es.ask()
        losses = []

        for theta in candidates:
            loss = evaluate_candidate(net, theta, train_samples, trace_alpha, count_weight)
            losses.append(loss)

        es.tell(candidates, losses)

        best_generation_loss = min(losses)
        candidate_theta = np.asarray(es.result.xbest).copy()
        validation_loss = evaluate_candidate(net, candidate_theta, fixed_validation_set, trace_alpha, count_weight)

        history["train_loss"].append(best_generation_loss)
        history["validation_loss"].append(validation_loss)

        generations_run = generation + 1

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_theta = candidate_theta.copy()
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if history_file is not None:
            save_history_row(
                history_file,
                configuration_id,
                run_id,
                stage,
                seed,
                generation + 1,
                best_generation_loss,
                validation_loss,
                best_validation_loss,
                es.sigma
            )

        if verbose:
            print(f"Generation {generation + 1:03d} | train={best_generation_loss:.6f} | validation={validation_loss:.6f} | sigma={es.sigma:.6f}")

        if es.stop():
            if verbose:
                print(f"CMA-ES stopping: {es.stop()}")
            break

        if generations_without_improvement >= patience:
            if verbose:
                print(f"Early stopping after {patience} generations without validation improvement")
            break

    set_parameters(net, best_validation_theta)
    net.reset_state()

    validation_metrics = evaluate_test_metrics(net, validation_loader, trace_alpha, count_weight)
    silent_metrics = test_silent_baseline(net, validation_loader, trace_alpha, count_weight)

    info = {
        "generations_run": generations_run,
        "best_training_fitness": float(es.result.fbest),
        "best_validation_loss": float(best_validation_loss)
    }

    return history, validation_metrics, silent_metrics, info


def make_grid(base_params, grid):
    keys = list(grid.keys())
    configs = []

    for values in product(*grid.values()):
        params = base_params.copy()
        params.update(dict(zip(keys, values)))
        configs.append(params)

    return configs


def run_experiment(local_dir, params, seed, stage, output_dir, evaluate_test=False):
    configuration_id = make_configuration_id(stage, params)
    run_id = make_run_id(configuration_id, seed)

    history_dir = output_dir / "history"
    history_file = history_dir / f"{run_id}.csv"

    print()
    print(f"[START] {run_id}", flush=True)
    print(f"Configuration: {configuration_id}", flush=True)
    print(f"Seed: {seed}", flush=True)
    print(f"Parameters: {params}", flush=True)
    print(flush=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    train_loader, validation_loader, test_loader = make_loaders(local_dir, seed)

    model = build_model(params)

    _, validation_metrics, silent_validation_metrics, info = train(
        model,
        train_loader,
        validation_loader,
        generations=MAX_GENERATIONS,
        patience=EARLY_STOPPING_PATIENCE,
        sigma=params["sigma"],
        samples_per_generation=params["samples_per_generation"],
        population_size=params["population_size"],
        validation_samples=VALIDATION_SAMPLES,
        trace_alpha=params["trace_alpha"],
        count_weight=params["count_weight"],
        seed=seed,
        verbose=False,
        configuration_id=configuration_id,
        run_id=run_id,
        stage=stage,
        history_file=history_file
    )

    if evaluate_test:
        metrics = evaluate_test_metrics(model, test_loader, params["trace_alpha"], params["count_weight"])
        silent_metrics = test_silent_baseline(model, test_loader, params["trace_alpha"], params["count_weight"])
        split = "test"
    else:
        metrics = validation_metrics
        silent_metrics = silent_validation_metrics
        split = "validation"

    result = {
        "configuration_id": configuration_id,
        "run_id": run_id,
        "history_file": str(history_file),
        "stage": stage,
        "split": split,
        "seed": seed,
        "count_weight": params["count_weight"],
        "trace_alpha": params["trace_alpha"],
        "sigma": params["sigma"],
        "samples_per_generation": params["samples_per_generation"],
        "population_size": params["population_size"],
        "decay": params["decay"],
        "threshold": params["threshold"],
        "max_generations": MAX_GENERATIONS,
        "generations_run": info["generations_run"],
        "best_training_fitness": info["best_training_fitness"],
        "best_validation_loss": info["best_validation_loss"],
        "loss": metrics["loss"],
        "trace_loss": metrics["trace_loss"],
        "count_loss": metrics["count_loss"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "precision_1": metrics["precision_1"],
        "recall_1": metrics["recall_1"],
        "f1_1": metrics["f1_1"],
        "precision_2": metrics["precision_2"],
        "recall_2": metrics["recall_2"],
        "f1_2": metrics["f1_2"],
        "input_spikes": metrics["input_spikes"],
        "output_spikes": metrics["output_spikes"],
        "spike_ratio": metrics["spike_ratio"],
        "silent_loss": silent_metrics["loss"],
        "silent_trace_loss": silent_metrics["trace_loss"],
        "silent_count_loss": silent_metrics["count_loss"]
    }

    print()
    print(f"[FINISH] {run_id}", flush=True)
    print(f"Configuration: {configuration_id}", flush=True)
    print(f"Generations: {info['generations_run']}", flush=True)
    print(f"{split.capitalize()} loss: {metrics['loss']:.6f}", flush=True)
    print(f"Exact F1: {metrics['f1']:.4f}", flush=True)
    print(f"+/-2 F1: {metrics['f1_2']:.4f}", flush=True)
    print(f"History: {history_file}", flush=True)
    print(flush=True)

    return result


def save_result(result, results_file):
    file_exists = results_file.exists()

    with open(results_file, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)



def cast_result_row(row):
    result = dict(row)

    integer_fields = {
        "seed",
        "samples_per_generation",
        "population_size",
        "max_generations",
        "generations_run"
    }

    float_fields = {
        "count_weight",
        "trace_alpha",
        "sigma",
        "decay",
        "threshold",
        "best_training_fitness",
        "best_validation_loss",
        "loss",
        "trace_loss",
        "count_loss",
        "precision",
        "recall",
        "f1",
        "precision_1",
        "recall_1",
        "f1_1",
        "precision_2",
        "recall_2",
        "f1_2",
        "input_spikes",
        "output_spikes",
        "spike_ratio",
        "silent_loss",
        "silent_trace_loss",
        "silent_count_loss"
    }

    for field in integer_fields:
        if field in result and result[field] not in (None, ""):
            result[field] = int(float(result[field]))

    for field in float_fields:
        if field in result and result[field] not in (None, ""):
            result[field] = float(result[field])

    return result


def load_completed_results(results_file):
    if not results_file.exists():
        return {}

    completed = {}

    with open(results_file, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if not row.get("run_id"):
                continue

            result = cast_result_row(row)
            completed[result["run_id"]] = result

    return completed


def get_run_status(run_id, output_dir, completed_results):
    if run_id in completed_results:
        return "complete"

    history_file = output_dir / "history" / f"{run_id}.csv"

    if history_file.exists():
        return "incomplete"

    return "not_started"

def save_configuration(configuration_id, stage, params, configuration_file):
    file_exists = configuration_file.exists()

    if file_exists:
        with open(configuration_file, "r", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["configuration_id"] == configuration_id:
                    return

    result = {
        "configuration_id": configuration_id,
        "stage": stage,
        "count_weight": params["count_weight"],
        "trace_alpha": params["trace_alpha"],
        "sigma": params["sigma"],
        "samples_per_generation": params["samples_per_generation"],
        "population_size": params["population_size"],
        "decay": params["decay"],
        "threshold": params["threshold"]
    }

    with open(configuration_file, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CONFIGURATION_FIELDS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


def select_best_config(results, metric=SELECTION_METRIC, required_runs=None):
    grouped_results = {}

    for result in results:
        key = tuple(result[name] for name in PARAMETER_KEYS)

        if key not in grouped_results:
            grouped_results[key] = []

        grouped_results[key].append(result)

    best_key = None
    best_mean = -np.inf
    best_std = 0.0

    for key, group in grouped_results.items():
        if required_runs is not None and len(group) != required_runs:
            continue

        values = [result[metric] for result in group]
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))

        if mean_value > best_mean:
            best_mean = mean_value
            best_std = std_value
            best_key = key

    if best_key is None:
        raise RuntimeError("No complete configuration had results for all required seeds")

    best_params = dict(zip(PARAMETER_KEYS, best_key))

    print()
    print(f"Best mean {metric}: {best_mean:.6f} +/- {best_std:.6f}")
    print(f"Best parameters: {best_params}")
    print()

    return best_params


def run_stage(stage_name, local_dir, base_params, grid, seeds, results_file, configuration_file, output_dir, max_workers):
    configs = make_grid(base_params, grid)
    total_runs = len(configs) * len(seeds)
    completed_results = load_completed_results(results_file)

    print()
    print(f"{stage_name}")
    print(f"Configurations: {len(configs)}")
    print(f"Seeds: {len(seeds)}")
    print(f"Runs: {total_runs}")
    print()

    results = []
    jobs = []
    skipped = 0
    incomplete = 0

    for params in configs:
        configuration_id = make_configuration_id(stage_name, params)
        save_configuration(configuration_id, stage_name, params, configuration_file)

        for seed in seeds:
            run_id = make_run_id(configuration_id, seed)
            status = get_run_status(run_id, output_dir, completed_results)

            if status == "complete":
                skipped += 1
                results.append(completed_results[run_id])
                print(f"[SKIP COMPLETE] {run_id}", flush=True)
            else:
                if status == "incomplete":
                    incomplete += 1
                    print(f"[RERUN INCOMPLETE] {run_id}", flush=True)

                jobs.append((params, seed))

    print()
    print(f"Already complete: {skipped}/{total_runs}")
    print(f"Incomplete and being rerun: {incomplete}")
    print(f"Remaining runs this launch: {len(jobs)}")
    print()

    if not jobs:
        print(f"[STAGE COMPLETE] {stage_name} - nothing left to run", flush=True)
        return results

    if max_workers == 1:
        for i, (params, seed) in enumerate(jobs, start=1):
            result = run_experiment(local_dir, params, seed, stage_name, output_dir)
            results.append(result)
            save_result(result, results_file)

            print(f"[SAVED] {result['run_id']} | {i}/{len(jobs)} remaining-run jobs", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for params, seed in jobs:
                future = executor.submit(run_experiment, local_dir, params, seed, stage_name, output_dir, False)
                futures[future] = (params, seed)

            completed_this_launch = 0

            for future in as_completed(futures):
                params, seed = futures[future]
                configuration_id = make_configuration_id(stage_name, params)
                run_id = make_run_id(configuration_id, seed)

                try:
                    result = future.result()
                except Exception as error:
                    print()
                    print(f"[FAILED] {run_id}", flush=True)
                    print(f"Error: {error}", flush=True)
                    print("This run will be treated as incomplete and rerun next time.", flush=True)
                    print()
                    continue

                completed_this_launch += 1
                results.append(result)
                save_result(result, results_file)

                print(f"[SAVED] {result['run_id']} | {completed_this_launch}/{len(jobs)} remaining-run jobs", flush=True)

    return results

def run_final_tests(local_dir, params, seeds, results_file, configuration_file, output_dir, max_workers):
    print()
    print("Final held-out test")
    print(f"Runs: {len(seeds)}")
    print()

    results = []
    jobs = []
    stage_name = "final_test"
    completed_results = load_completed_results(results_file)

    configuration_id = make_configuration_id(stage_name, params)
    save_configuration(configuration_id, stage_name, params, configuration_file)

    for seed in seeds:
        run_id = make_run_id(configuration_id, seed)
        status = get_run_status(run_id, output_dir, completed_results)

        if status == "complete":
            results.append(completed_results[run_id])
            print(f"[SKIP COMPLETE] {run_id}", flush=True)
        else:
            if status == "incomplete":
                print(f"[RERUN INCOMPLETE] {run_id}", flush=True)

            jobs.append(seed)

    print()
    print(f"Already complete: {len(results)}/{len(seeds)}")
    print(f"Remaining final-test runs this launch: {len(jobs)}")
    print()

    if not jobs:
        print("[FINAL TEST COMPLETE] Nothing left to run", flush=True)
        return results

    if max_workers == 1:
        for i, seed in enumerate(jobs, start=1):
            result = run_experiment(local_dir, params, seed, stage_name, output_dir, True)
            results.append(result)
            save_result(result, results_file)

            print(f"[SAVED] {result['run_id']} | {i}/{len(jobs)} remaining final-test runs", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for seed in jobs:
                future = executor.submit(run_experiment, local_dir, params, seed, stage_name, output_dir, True)
                futures[future] = seed

            completed_this_launch = 0

            for future in as_completed(futures):
                seed = futures[future]
                run_id = make_run_id(configuration_id, seed)

                try:
                    result = future.result()
                except Exception as error:
                    print()
                    print(f"[FAILED] {run_id}", flush=True)
                    print(f"Error: {error}", flush=True)
                    print("This run will be treated as incomplete and rerun next time.", flush=True)
                    print()
                    continue

                completed_this_launch += 1
                results.append(result)
                save_result(result, results_file)

                print(f"[SAVED] {result['run_id']} | {completed_this_launch}/{len(jobs)} remaining final-test runs", flush=True)

    return results

def print_final_summary(results):
    print()
    print("Final test summary")
    print()

    for metric in ["loss", "trace_loss", "count_loss", "precision", "recall", "f1", "f1_1", "f1_2", "spike_ratio"]:
        values = [result[metric] for result in results]

        print(f"{metric}: {np.mean(values):.6f} +/- {np.std(values):.6f}")


if __name__ == "__main__":
    home = pathlib.Path("~").expanduser()
    local_dir = home / "data" / "URBAN-SED"

    output_dir = pathlib.Path("GridSearchOuts")
    history_dir = output_dir / "history"

    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "results.csv"
    configuration_file = output_dir / "configurations.csv"

    base_params = {
        "count_weight": 0.03,
        "trace_alpha": 0.9,
        "sigma": 0.05,
        "samples_per_generation": 8,
        "population_size": 17,
        "decay": 0.9,
        "threshold": 1.0
    }

    stage_1_grid = {
        "count_weight": [0.01, 0.03, 0.05],
        "sigma": [0.05, 0.1, 0.15],
        "samples_per_generation": [8, 16, 24],
        "population_size": [17, 32, 48]
    }

    stage_2_grid = {
        "decay": [0.8, 0.9, 0.95, 0.99],
        "threshold": [0.5, 0.75, 1.0, 1.25]
    }

    stage_3_grid = {
        "trace_alpha": [0.8, 0.9, 0.95],
        "count_weight": [0.02, 0.03, 0.04, 0.05]
    }

    stage_1_runs = len(make_grid(base_params, stage_1_grid)) * len(SEEDS)
    stage_2_runs = len(make_grid(base_params, stage_2_grid)) * len(SEEDS)
    stage_3_runs = len(make_grid(base_params, stage_3_grid)) * len(SEEDS)

    search_runs = stage_1_runs + stage_2_runs + stage_3_runs
    total_runs = search_runs + len(SEEDS)

    print(f"Parallel workers: {MAX_WORKERS}")
    print(f"Seeds: {SEEDS}")
    print(f"Stage 1 runs: {stage_1_runs}")
    print(f"Stage 2 runs: {stage_2_runs}")
    print(f"Stage 3 runs: {stage_3_runs}")
    print(f"Grid-search runs: {search_runs}")
    print(f"Final test runs: {len(SEEDS)}")
    print(f"Total runs: {total_runs}")
    print(f"Results file: {results_file}")
    print(f"Configuration file: {configuration_file}")
    print(f"History directory: {history_dir}")

    stage_1_results = run_stage(
        "stage_1",
        local_dir,
        base_params,
        stage_1_grid,
        SEEDS,
        results_file,
        configuration_file,
        output_dir,
        MAX_WORKERS
    )

    best_stage_1 = select_best_config(
        stage_1_results,
        required_runs=len(SEEDS)
    )

    stage_2_results = run_stage(
        "stage_2",
        local_dir,
        best_stage_1,
        stage_2_grid,
        SEEDS,
        results_file,
        configuration_file,
        output_dir,
        MAX_WORKERS
    )

    best_stage_2 = select_best_config(
        stage_2_results,
        required_runs=len(SEEDS)
    )

    stage_3_results = run_stage(
        "stage_3",
        local_dir,
        best_stage_2,
        stage_3_grid,
        SEEDS,
        results_file,
        configuration_file,
        output_dir,
        MAX_WORKERS
    )

    best_params = select_best_config(
        stage_3_results,
        required_runs=len(SEEDS)
    )

    print()
    print("Selected parameters:")
    print(best_params)
    print()

    final_results = run_final_tests(
        local_dir,
        best_params,
        SEEDS,
        results_file,
        configuration_file,
        output_dir,
        MAX_WORKERS
    )

    print_final_summary(final_results)