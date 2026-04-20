import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

MODE = ""

def load_data():
    global MODE
    MODE = input("raw_data or Processed Data: (r/p)\n")

    # Select which data to visualize
    if (MODE == "r"):
        conda_dir = Path("data/")
    if (MODE == "p"):
        conda_dir = Path("data/processed")

    # Find the paths to the data
    if conda_dir.exists() and conda_dir.is_dir():
        csv_files = sorted(conda_dir.glob("*.csv"))
    else:
        csv_files = sorted(Path(".").glob("*CONDA*.csv"))

    # Collects CONDA Data
    conda_data = {file.stem: pd.read_csv(file) for file in csv_files}
    
    return(conda_data)


def show_times(data):
    for name, df in data.items():
        if "chatTime" in df.columns:
            print(f"\n{name} chatTime:")
            print(f"  Min: {df['chatTime'].min()}")
            print(f"  Max: {df['chatTime'].max()}")
            print(f"  Avg: {df['chatTime'].mean():.2f}")

            

def show_toxicity_over_time(data):
    # Combine train and valid sets
    dfs = []
    for name in ["CONDA_train", "CONDA_train_cleaned"]:
        if name in data:
            dfs.append(data[name])
            break
    for name in ["CONDA_valid", "CONDA_valid_cleaned"]:
        if name in data:
            dfs.append(data[name])
            break

    if not dfs:
        print("No train/valid data found.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    # Mark toxic (E or I) vs non-toxic
    combined["toxic"] = combined["intentClass"].isin(["E", "I"]).astype(int)

    # Bin chatTime into intervals
    combined["timeBin"] = pd.cut(combined["chatTime"], bins=20)
    grouped = combined.groupby("timeBin", observed=True)["toxic"].mean()

    # Plot toxicity rate over time
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(grouped)), grouped.values * 100, marker="o")
    plt.xticks(range(len(grouped)), [f"{b.left:.0f}-{b.right:.0f}" for b in grouped.index], rotation=45, ha="right")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    plt.xlabel("Game Time (chatTime)")
    plt.ylabel("Toxicity Rate (E + I)")
    plt.title("Toxicity Over Game Time (Train + Valid)")
    plt.tight_layout()
    plt.show()


def show_time_distribution(data):
    # Combine train and valid sets
    dfs = []
    for name in ["CONDA_train", "CONDA_train_cleaned"]:
        if name in data:
            dfs.append(data[name])
            break
    for name in ["CONDA_valid", "CONDA_valid_cleaned"]:
        if name in data:
            dfs.append(data[name])
            break

    if not dfs:
        print("No train/valid data found.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    times = combined["chatTime"]

    avg = times.mean()
    upper_avg = times[times >= avg].mean()
    lower_avg = times[times < avg].mean()

    plt.figure(figsize=(10, 5))
    plt.hist(times, bins=40, edgecolor="black", alpha=0.7)
    plt.axvline(avg, color="red", linestyle="--", linewidth=2, label=f"Average: {avg:.0f}")
    plt.axvline(upper_avg, color="orange", linestyle="--", linewidth=2, label=f"Upper Avg: {upper_avg:.0f}")
    plt.axvline(lower_avg, color="blue", linestyle="--", linewidth=2, label=f"Lower Avg: {lower_avg:.0f}")
    plt.xlabel("Game Time (chatTime)")
    plt.ylabel("Message Count")
    plt.title("Chat Time Distribution (Train + Valid)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_target_label_balance(data):
    dfs = []
    for name in ["CONDA_train", "CONDA_train_cleaned"]:
        if name in data:
            dfs.append(data[name])
            break

    if not dfs:
        print("No train data found.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    counts = combined["intentClass"].value_counts().sort_index()

    # Print counts to terminal
    print("\nIntent Class Distribution:")
    for label, count in counts.items():
        print(f"  {label}: {count}")
    print(f"  Total: {counts.sum()}")

    # Bar graph
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values, color=["red", "orange", "steelblue", "gray"], edgecolor="black")
    for i, (label, count) in enumerate(counts.items()):
        plt.text(i, count + counts.max() * 0.01, str(count), ha="center", fontweight="bold")
    plt.xlabel("Intent Class")
    plt.ylabel("Count")
    plt.title("Target Label Balance (Intent Classes)")
    plt.tight_layout()
    plt.show()


def run_visualization():
    # Fetch Data
    data = load_data()
    show_times(data)
    show_toxicity_over_time(data)
    show_time_distribution(data)
    show_target_label_balance(data)

