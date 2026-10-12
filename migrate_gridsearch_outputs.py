from pathlib import Path

output_dir = Path("GridSearchOuts")

results_target = output_dir / "results.csv"
configurations_target = output_dir / "configurations.csv"

results_files = sorted(output_dir.glob("cmaes_grid_search_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
configuration_files = sorted(output_dir.glob("configurations_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)

if not output_dir.exists():
    raise FileNotFoundError("GridSearchOuts does not exist")

if results_target.exists():
    print(f"Already exists: {results_target}")
else:
    if not results_files:
        print("No timestamped grid-search results CSV found")
    else:
        source = results_files[0]
        source.rename(results_target)
        print(f"Renamed {source.name} -> {results_target.name}")

if configurations_target.exists():
    print(f"Already exists: {configurations_target}")
else:
    if not configuration_files:
        print("No timestamped configurations CSV found")
    else:
        source = configuration_files[0]
        source.rename(configurations_target)
        print(f"Renamed {source.name} -> {configurations_target.name}")

history_dir = output_dir / "history"

if history_dir.exists():
    print(f"History directory already correct: {history_dir}")
else:
    print("No history directory found")