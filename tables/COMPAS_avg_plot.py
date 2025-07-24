import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def plot_combined_from_csv(csv_path1="results/Biased-COMPAS/Compas_avg.csv", csv_path2="results/Fairwashed-COMPAS/Compas_avg.csv", id1='Biased', id2='Fairwashed', output_path='results/plots/Figure9.pdf', figsize=(9, 4)):
    # Load first CSV
    with open(csv_path1, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        feature_order = reader[0]
        vals1 = [float(x) for x in reader[1]]

    # Load second CSV and build a feature-to-value mapping
    with open(csv_path2, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        feature_names_2 = reader[0]
        vals2_raw = [float(x) for x in reader[1]]
        feature_to_val_2 = dict(zip(feature_names_2, vals2_raw))

    # Reorder vals2 according to the feature_order
    try:
        vals2 = [feature_to_val_2[name] for name in feature_order]
    except KeyError as e:
        raise ValueError(f"Feature '{e.args[0]}' from first CSV not found in second CSV.")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")


    # Plot
    fig = plt.figure(figsize=figsize)
    pos = np.arange(len(feature_order))
    bar_width = 0.35

    plt.barh(pos - bar_width/2, vals1, height=bar_width, color='orange', label=id1)
    plt.barh(pos + bar_width/2, vals2, height=bar_width, color='skyblue', label=id2)

    plt.yticks(pos, feature_order)
    plt.title("Average Absolute Attribution")
    plt.xlabel("Relevance")
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_path, transparent=False, pad_inches=0, bbox_inches='tight', orientation='portrait')
    return fig

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot average absolute attributions from two CSV files.")
    parser.add_argument("--csv_path1", type=str, default="results/Biased-COMPAS/Compas_avg.csv", help="Path to the first CSV file.")
    parser.add_argument("--csv_path2", type=str, default="results/Fairwashed-COMPAS/Compas_avg.csv", help="Path to the second CSV file.")
    parser.add_argument("--id1", type=str, default='Biased')
    parser.add_argument("--id2", type=str, default='Fairwashed')
    parser.add_argument("--output_path", type=str, default='results/plots/Figure9.pdf', help="Output path for the plot.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(9, 4), help="Figure size in inches.")

    args = parser.parse_args()
    
    plot_combined_from_csv(args.csv_path1, args.csv_path2, args.id1, args.id2, args.output_path, tuple(args.figsize))
    print(f"Plot saved to {args.output_path}")