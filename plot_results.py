import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = pathlib.Path("GridSearchOuts")
RESULTS_FILE = OUTPUT_DIR / "results.csv"
HISTORY_DIR = OUTPUT_DIR / "history"
PLOT_DIR = OUTPUT_DIR / "plots"

EXPECTED_SEEDS = [101, 202, 303, 404, 505]
EXPECTED_RUNS_PER_CONFIG = len(EXPECTED_SEEDS)

SELECTION_METRIC = "f1_2"

PARAMETER_NAMES = {
    "count_weight": "Count weight",
    "trace_alpha": "Trace alpha",
    "sigma": "Initial sigma",
    "samples_per_generation": "Samples / generation",
    "population_size": "Population size",
    "decay": "Decay",
    "threshold": "Threshold"
}

STAGE_PARAMETERS = {
    "stage_1": ["count_weight", "sigma", "samples_per_generation", "population_size"],
    "stage_2": ["decay", "threshold"],
    "stage_3": ["trace_alpha", "count_weight"]
}


def load_results():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Could not find {RESULTS_FILE}")

    results = pd.read_csv(RESULTS_FILE)

    print(f"Loaded {len(results)} completed runs")

    return results


def get_validation_results(results):
    return results[results["split"] == "validation"].copy()


def get_complete_results(results):
    validation_results = get_validation_results(results)

    counts = validation_results.groupby("configuration_id")["seed"].nunique()
    complete_ids = counts[counts == EXPECTED_RUNS_PER_CONFIG].index

    complete = validation_results[validation_results["configuration_id"].isin(complete_ids)].copy()

    print(f"Complete validation configurations: {len(complete_ids)}")
    print(f"Complete validation runs: {len(complete)}")

    return complete


def aggregate_configurations(results):
    numeric_columns = [
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
        "spike_ratio",
        "generations_run"
    ]

    parameter_columns = [
        "stage",
        "count_weight",
        "trace_alpha",
        "sigma",
        "samples_per_generation",
        "population_size",
        "decay",
        "threshold"
    ]

    aggregation = {}

    for column in parameter_columns:
        aggregation[column] = "first"

    for column in numeric_columns:
        aggregation[column] = ["mean", "std"]

    grouped = results.groupby("configuration_id").agg(aggregation)

    grouped.columns = ["_".join(column).rstrip("_") for column in grouped.columns]
    grouped = grouped.reset_index()

    return grouped


def short_config_label(row):
    if row["stage_first"] == "stage_1":
        return f"cw={row['count_weight_first']:g}, sig={row['sigma_first']:g}, spg={int(row['samples_per_generation_first'])}, pop={int(row['population_size_first'])}"

    if row["stage_first"] == "stage_2":
        return f"dec={row['decay_first']:g}, th={row['threshold_first']:g}"

    if row["stage_first"] == "stage_3":
        return f"ta={row['trace_alpha_first']:g}, cw={row['count_weight_first']:g}"

    return row["configuration_id"]


def plot_top_configurations(aggregated, stage, top_n=10):
    stage_data = aggregated[aggregated["stage_first"] == stage].copy()

    if stage_data.empty:
        return

    stage_data = stage_data.sort_values(f"{SELECTION_METRIC}_mean", ascending=False).head(top_n)
    stage_data = stage_data.iloc[::-1]

    labels = [short_config_label(row) for _, row in stage_data.iterrows()]

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.barh(labels, stage_data[f"{SELECTION_METRIC}_mean"], xerr=stage_data[f"{SELECTION_METRIC}_std"], capsize=4)

    ax.set_xlabel("Mean ±2 F1")
    ax.set_ylabel("Configuration")
    ax.set_title(f"{stage}: top {top_n} configurations")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{stage}_top_{top_n}_configurations.png", dpi=300)
    plt.show()
    plt.close(fig)


def plot_parameter_effect(results, stage, parameter, metric="f1_2"):
    stage_data = results[results["stage"] == stage].copy()

    if stage_data.empty:
        return

    summary = stage_data.groupby(parameter)[metric].agg(["mean", "std", "count"]).reset_index()
    summary = summary.sort_values(parameter)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(summary[parameter], summary["mean"], yerr=summary["std"], marker="o", capsize=4)

    ax.set_xlabel(PARAMETER_NAMES.get(parameter, parameter))
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"{stage}: effect of {PARAMETER_NAMES.get(parameter, parameter)} on {metric}")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{stage}_{parameter}_{metric}.png", dpi=300)
    plt.show()
    plt.close(fig)


def plot_parameter_effects(results):
    for stage, parameters in STAGE_PARAMETERS.items():
        for parameter in parameters:
            plot_parameter_effect(results, stage, parameter, "f1_2")
            plot_parameter_effect(results, stage, parameter, "trace_loss")
            plot_parameter_effect(results, stage, parameter, "spike_ratio")


def plot_exact_vs_tolerant_f1(results):
    fig, ax = plt.subplots(figsize=(7, 6))

    for stage in ["stage_1", "stage_2", "stage_3"]:
        stage_data = results[results["stage"] == stage]

        if not stage_data.empty:
            ax.scatter(stage_data["f1"], stage_data["f1_2"], alpha=0.55, label=stage)

    limits = [
        min(results["f1"].min(), results["f1_2"].min()),
        max(results["f1"].max(), results["f1_2"].max())
    ]

    ax.plot(limits, limits, linestyle="--", alpha=0.6)

    ax.set_xlabel("Exact F1")
    ax.set_ylabel("±2 F1")
    ax.set_title("Exact vs temporally tolerant reconstruction")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exact_vs_tolerant_f1.png", dpi=300)
    plt.show()
    plt.close(fig)


def plot_spike_ratio_vs_f1(results):
    fig, ax = plt.subplots(figsize=(7, 6))

    for stage in ["stage_1", "stage_2", "stage_3"]:
        stage_data = results[results["stage"] == stage]

        if not stage_data.empty:
            ax.scatter(stage_data["spike_ratio"], stage_data["f1_2"], alpha=0.55, label=stage)

    ax.axvline(1.0, linestyle="--", alpha=0.6)

    ax.set_xlabel("Output / input spike ratio")
    ax.set_ylabel("±2 F1")
    ax.set_title("Spike-count preservation vs reconstruction")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "spike_ratio_vs_f1_2.png", dpi=300)
    plt.show()
    plt.close(fig)


def plot_trace_loss_vs_f1(results):
    fig, ax = plt.subplots(figsize=(7, 6))

    for stage in ["stage_1", "stage_2", "stage_3"]:
        stage_data = results[results["stage"] == stage]

        if not stage_data.empty:
            ax.scatter(stage_data["trace_loss"], stage_data["f1_2"], alpha=0.55, label=stage)

    ax.set_xlabel("Trace loss")
    ax.set_ylabel("±2 F1")
    ax.set_title("Trace loss vs spike-event reconstruction")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "trace_loss_vs_f1_2.png", dpi=300)
    plt.show()
    plt.close(fig)


def plot_precision_recall(results):
    fig, ax = plt.subplots(figsize=(7, 6))

    for stage in ["stage_1", "stage_2", "stage_3"]:
        stage_data = results[results["stage"] == stage]

        if not stage_data.empty:
            ax.scatter(stage_data["precision_2"], stage_data["recall_2"], alpha=0.55, label=stage)

    ax.set_xlabel("±2 precision")
    ax.set_ylabel("±2 recall")
    ax.set_title("Precision-recall trade-off")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "precision_vs_recall_2.png", dpi=300)
    plt.show()
    plt.close(fig)


def plot_generation_distribution(results):
    fig, ax = plt.subplots(figsize=(8, 5))

    stages = []
    data = []

    for stage in ["stage_1", "stage_2", "stage_3"]:
        values = results.loc[results["stage"] == stage, "generations_run"]

        if len(values) > 0:
            stages.append(stage)
            data.append(values.to_numpy())

    ax.boxplot(data, tick_labels=stages)

    ax.axhline(250, linestyle="--", alpha=0.6)

    ax.set_xlabel("Stage")
    ax.set_ylabel("Generations run")
    ax.set_title("CMA-ES generations before stopping")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "generations_run_distribution.png", dpi=300)
    plt.show()
    plt.close(fig)


def get_best_configuration(results, stage):
    stage_data = results[results["stage"] == stage].copy()

    if stage_data.empty:
        return None

    means = stage_data.groupby("configuration_id")[SELECTION_METRIC].mean()

    return means.idxmax()


def load_configuration_histories(results, configuration_id):
    config_runs = results[results["configuration_id"] == configuration_id]

    histories = []

    for _, run in config_runs.iterrows():
        history_file = pathlib.Path(run["history_file"])

        if not history_file.is_absolute():
            if history_file.exists():
                pass
            elif (OUTPUT_DIR.parent / history_file).exists():
                history_file = OUTPUT_DIR.parent / history_file
            elif (HISTORY_DIR / history_file.name).exists():
                history_file = HISTORY_DIR / history_file.name

        if not history_file.exists():
            print(f"History not found: {history_file}")
            continue

        history = pd.read_csv(history_file)
        histories.append(history)

    return histories


def combine_histories(histories):
    if not histories:
        return None

    combined = pd.concat(histories, ignore_index=True)

    summary = combined.groupby("generation").agg(
        train_mean=("train_loss", "mean"),
        train_std=("train_loss", "std"),
        validation_mean=("validation_loss", "mean"),
        validation_std=("validation_loss", "std"),
        best_validation_mean=("best_validation_loss", "mean"),
        best_validation_std=("best_validation_loss", "std"),
        sigma_mean=("sigma", "mean"),
        sigma_std=("sigma", "std"),
        runs=("run_id", "nunique")
    ).reset_index()

    summary = summary.fillna(0.0)

    return summary


def plot_best_convergence(results, stage):
    configuration_id = get_best_configuration(results, stage)

    if configuration_id is None:
        return

    histories = load_configuration_histories(results, configuration_id)
    summary = combine_histories(histories)

    if summary is None:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(summary["generation"], summary["train_mean"], label="Train")
    ax.fill_between(summary["generation"], summary["train_mean"] - summary["train_std"], summary["train_mean"] + summary["train_std"], alpha=0.2)

    ax.plot(summary["generation"], summary["validation_mean"], label="Validation")
    ax.fill_between(summary["generation"], summary["validation_mean"] - summary["validation_std"], summary["validation_mean"] + summary["validation_std"], alpha=0.2)

    ax.plot(summary["generation"], summary["best_validation_mean"], label="Best validation")
    ax.fill_between(summary["generation"], summary["best_validation_mean"] - summary["best_validation_std"], summary["best_validation_mean"] + summary["best_validation_std"], alpha=0.2)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Loss")
    ax.set_title(f"{stage}: convergence of best configuration")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{stage}_best_convergence.png", dpi=300)
    plt.show()
    plt.close(fig)


def plot_best_sigma(results, stage):
    configuration_id = get_best_configuration(results, stage)

    if configuration_id is None:
        return

    histories = load_configuration_histories(results, configuration_id)
    summary = combine_histories(histories)

    if summary is None:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(summary["generation"], summary["sigma_mean"])
    ax.fill_between(summary["generation"], summary["sigma_mean"] - summary["sigma_std"], summary["sigma_mean"] + summary["sigma_std"], alpha=0.2)

    ax.set_xlabel("Generation")
    ax.set_ylabel("CMA-ES sigma")
    ax.set_title(f"{stage}: sigma adaptation for best configuration")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{stage}_best_sigma.png", dpi=300)
    plt.show()
    plt.close(fig)


def plot_seed_variability(results, aggregated, stage, top_n=5):
    stage_aggregated = aggregated[aggregated["stage_first"] == stage].copy()

    if stage_aggregated.empty:
        return

    best_ids = stage_aggregated.sort_values(f"{SELECTION_METRIC}_mean", ascending=False).head(top_n)["configuration_id"].tolist()

    plot_data = results[results["configuration_id"].isin(best_ids)].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    positions = np.arange(len(best_ids))

    for index, configuration_id in enumerate(best_ids):
        values = plot_data.loc[plot_data["configuration_id"] == configuration_id, SELECTION_METRIC].to_numpy()
        x = np.full(len(values), index)

        ax.scatter(x, values, alpha=0.75)

        if len(values) > 0:
            ax.scatter(index, np.mean(values), marker="x", s=100)

    labels = []

    for configuration_id in best_ids:
        row = stage_aggregated[stage_aggregated["configuration_id"] == configuration_id].iloc[0]
        labels.append(short_config_label(row))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("±2 F1")
    ax.set_title(f"{stage}: seed variability of top {top_n} configurations")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{stage}_top_{top_n}_seed_variability.png", dpi=300)
    plt.show()
    plt.close(fig)


def plot_final_test(results):
    final_results = results[(results["stage"] == "final_test") & (results["split"] == "test")].copy()

    if final_results.empty:
        print("No final test results yet")
        return

    metrics = ["f1", "f1_1", "f1_2"]
    labels = ["Exact", "±1", "±2"]

    means = [final_results[metric].mean() for metric in metrics]
    stds = [final_results[metric].std() for metric in metrics]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(labels, means, yerr=stds, capsize=5)

    ax.set_ylabel("F1")
    ax.set_title("Final held-out test reconstruction")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "final_test_f1.png", dpi=300)
    plt.show()
    plt.close(fig)


def print_best_configurations(aggregated):
    print()
    print("BEST CONFIGURATIONS")
    print("=" * 80)

    for stage in ["stage_1", "stage_2", "stage_3"]:
        stage_data = aggregated[aggregated["stage_first"] == stage].copy()

        if stage_data.empty:
            continue

        best = stage_data.sort_values(f"{SELECTION_METRIC}_mean", ascending=False).iloc[0]

        print()
        print(stage)
        print(f"Configuration: {best['configuration_id']}")
        print(f"Mean exact F1: {best['f1_mean']:.4f} +/- {best['f1_std']:.4f}")
        print(f"Mean +/-1 F1: {best['f1_1_mean']:.4f} +/- {best['f1_1_std']:.4f}")
        print(f"Mean +/-2 F1: {best['f1_2_mean']:.4f} +/- {best['f1_2_std']:.4f}")
        print(f"Mean trace loss: {best['trace_loss_mean']:.6f} +/- {best['trace_loss_std']:.6f}")
        print(f"Mean spike ratio: {best['spike_ratio_mean']:.4f} +/- {best['spike_ratio_std']:.4f}")

def plot_best_f1_comparison(aggregated):
    if aggregated.empty:
        return

    best_exact = aggregated.loc[aggregated["f1_mean"].idxmax()]
    best_1 = aggregated.loc[aggregated["f1_1_mean"].idxmax()]
    best_2 = aggregated.loc[aggregated["f1_2_mean"].idxmax()]

    selected = [
        ("Best exact F1", best_exact),
        ("Best ±1 F1", best_1),
        ("Best ±2 F1", best_2)
    ]

    labels = [item[0] for item in selected]

    exact_means = [item[1]["f1_mean"] for item in selected]
    f1_1_means = [item[1]["f1_1_mean"] for item in selected]
    f1_2_means = [item[1]["f1_2_mean"] for item in selected]

    exact_stds = [item[1]["f1_std"] for item in selected]
    f1_1_stds = [item[1]["f1_1_std"] for item in selected]
    f1_2_stds = [item[1]["f1_2_std"] for item in selected]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, exact_means, width, yerr=exact_stds, capsize=4, label="Exact F1")
    ax.bar(x, f1_1_means, width, yerr=f1_1_stds, capsize=4, label="±1 F1")
    ax.bar(x + width, f1_2_means, width, yerr=f1_2_stds, capsize=4, label="±2 F1")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean F1")
    ax.set_title("Comparison of configurations selected by exact and tolerant F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "best_exact_vs_tolerant_f1.png", dpi=300)
    plt.show()
    plt.close(fig)

    print()
    print("BEST F1 CONFIGURATION COMPARISON")
    print("=" * 80)

    for selection_name, row in selected:
        print()
        print(selection_name)
        print(f"Configuration: {row['configuration_id']}")
        print(f"Stage: {row['stage_first']}")
        print(f"Exact F1: {row['f1_mean']:.4f} +/- {row['f1_std']:.4f}")
        print(f"+/-1 F1: {row['f1_1_mean']:.4f} +/- {row['f1_1_std']:.4f}")
        print(f"+/-2 F1: {row['f1_2_mean']:.4f} +/- {row['f1_2_std']:.4f}")
        print(f"Trace loss: {row['trace_loss_mean']:.6f}")
        print(f"Spike ratio: {row['spike_ratio_mean']:.4f}")

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    complete_results = get_complete_results(results)
    aggregated = aggregate_configurations(complete_results)

    print_best_configurations(aggregated)

    for stage in ["stage_1", "stage_2", "stage_3"]:
        plot_top_configurations(aggregated, stage, top_n=10)
        plot_seed_variability(complete_results, aggregated, stage, top_n=5)
        plot_best_convergence(complete_results, stage)
        plot_best_sigma(complete_results, stage)

    plot_parameter_effects(complete_results)
    plot_exact_vs_tolerant_f1(complete_results)
    plot_spike_ratio_vs_f1(complete_results)
    plot_trace_loss_vs_f1(complete_results)
    plot_precision_recall(complete_results)
    plot_generation_distribution(complete_results)
    plot_best_f1_comparison(aggregated)

    plot_final_test(results)

    print()
    print(f"Plots saved to: {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    main()