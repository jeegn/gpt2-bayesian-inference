import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_inference_results(metrics: dict, times: dict, save_dir: str = "./figures"):
    os.makedirs(save_dir, exist_ok=True)

    # Normalize times based on fastest time
    fastest_time = min(times.values())
    relative_speed = {k: fastest_time / times[k] for k in times.keys()}

    metric_names = ["relative_speed", "accuracy", "log_loss", "brier_score", "ece_global", ]
    metric_labels = ["Inference Speed ↑", "Accuracy ↑", "Log Loss ↓", "Brier Score ↓", "ECE ↓", ]

    methods = list(metrics.keys())
    colors = ['gray', 'mediumseagreen', 'orchid', 'teal', 'orange', 'teal']

    # Extend metrics dict to include relative speed
    metrics_with_speed = {}
    for method in methods:
        metrics_with_speed[method] = metrics[method].copy()
        metrics_with_speed[method]["relative_speed"] = relative_speed[method]

    x = np.arange(len(metric_names)) * 0.8  # Spacing between metric groups
    width = 0.12
    offsets = np.linspace(-width * (len(methods) - 1) / 2, width * (len(methods) - 1) / 2, len(methods))

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, (method, offset, color) in enumerate(zip(methods, offsets, colors)):
        values = [metrics_with_speed[method][m] for m in metric_names]
        ax.bar(x + offset, values, width=width, label=method, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=0, ha='center', fontsize=16)
    ax.set_yscale('log')
    ax.set_ylabel('Metric Value (log scale)', fontsize=16)
    ax.set_title('Metrics on GoEmotions', fontsize=24)
    ax.legend(title='Inference Methods', fontsize=14, title_fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.42), ncol=3)
    ax.grid(True, axis='y', which='both', linestyle='--', linewidth=0.7, alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "performance_metrics_with_speed_log.png"))
    plt.close()

    print(f"Saved updated plot to {save_dir}/")

def plot_id_vs_ood_grouped(metrics_id: dict, metrics_ood: dict, save_dir: str = "./figures"):
    os.makedirs(save_dir, exist_ok=True)

    metric_names = ["accuracy", "log_loss", "brier_score", "ece_global"]
    display_names = ["Accuracy ↑", "Log Loss ↓", "Brier Score ↓", "ECE ↓"]

    methods = list(metrics_id.keys())

    method_styles = {
        "MAP": {"marker": "o", "color": "gray"},
        "Temp Scaling": {"marker": "s", "color": "orange"},
        "Laplace Approx": {"marker": "X", "color": "teal"},
        "MC Dropout": {"marker": "^", "color": "orchid"},
        "SGHMC": {"marker": "v", "color": "mediumseagreen"},
        "Deep Ensemble": {"marker": "P", "color": "blue"},
    }

    fig, axes = plt.subplots(1, len(metric_names), figsize=(8, 3))

    num_methods = len(methods)
    spacing = 0.5  # Spacing between methods

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        
        x_positions = np.arange(num_methods) * spacing  # Left side (ID)
        x_positions_ood = x_positions + (num_methods + 1) * spacing  # Right side (OOD)

        for i, method in enumerate(methods):
            style = method_styles.get(method, {"marker": "o", "color": "black"})

            id_value = metrics_id[method][metric]
            ood_value = metrics_ood[method][metric]

            # ID point
            ax.scatter(
                x_positions[i], id_value,
                marker=style["marker"], color=style["color"], s=120,
                label=method if idx == 0 else ""
            )
            # OOD point
            ax.scatter(
                x_positions_ood[i], ood_value,
                marker=style["marker"], color=style["color"], s=120
            )

        # Pretty X-ticks
        # all_positions = np.concatenate([x_positions, x_positions_ood])
        # all_labels = [m for m in methods] + [m for m in methods]
        # ax.set_xticks(all_positions)
        # ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=10)

        # Add separation line
        mid = (x_positions[-1] + x_positions_ood[0]) / 2
        ax.axvline(mid - spacing/2, color="black", linestyle="--", linewidth=1)

        # if idx == 0:
        ax.text(mid / 2, ax.get_ylim()[1] * 1.2, "ID", ha="center", fontsize=12)
        ax.text((mid + x_positions_ood[-1]) / 2, ax.get_ylim()[1] * 1.2, "OOD", ha="center", fontsize=12)

        ax.set_title(display_names[idx], fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="y", labelsize=12)
        # ax.set_yscale("log")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=num_methods, fontsize=12, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(save_dir, "id_ood_grouped_metrics.png"))
    plt.close()

    print(f"Saved ID/OOD grouped plots to {save_dir}/")

metrics = {
    "MAP": {
        "accuracy": 0.5043,
        "log_loss": 0.0921,
        "brier_score": 0.0239,
        "ece_global": 0.0138,
    },
    "Temp Scaling": {
        "accuracy": 0.4629,
        "log_loss": 0.0889,
        "brier_score": 0.0239,
        "ece_global": 0.0083,
    },
    "Laplace Approx": {
        "accuracy": 0.4968,
        "log_loss": 0.1277,
        "brier_score": 0.0279,
        "ece_global": 0.0226,
    },
    "MC Dropout": {
        "accuracy": 0.5001,
        "log_loss": 0.0925,
        "brier_score": 0.0241,
        "ece_global": 0.0140,
    },
    "SGHMC": {
        "accuracy": 0.4601,
        "log_loss": 0.0912,
        "brier_score": 0.0240,
        "ece_global": 0.0114,
    },
    
    # "Deep Ensemble": {
    #     "accuracy": 0.3228,
    #     "log_loss": 0.1297,
    #     "brier_score": 0.0336,
    #     "ece_global": 0.0161,
    # },
    
}

ood_metrics = {
    "MAP": {
        "accuracy": 0.2007,
        "log_loss": 1.0349,
        "brier_score": 0.2140,
        "ece_global": 0.1951,
    },
    "Laplace Approx": {
        "accuracy": 0.1115,
        "log_loss": 1.0496,
        "brier_score": 0.2221,
        "ece_global": 0.2099,
    },
    "MC Dropout": {
        "accuracy": 0.1514,
        "log_loss": 1.0757,
        "brier_score": 0.2185,
        "ece_global": 0.2015,
    },
    "SGHMC": {
        "accuracy": 0.2084,
        "log_loss": 0.9918,
        "brier_score": 0.2122,
        "ece_global": 0.1896,
    },
    # "Deep Ensemble": {
    #     "accuracy": 0.0324,
    #     "log_loss": 1.1130,
    #     "brier_score": 0.2263,
    #     "ece_global": 0.1993,
    # },
    "Temp Scaling": {
        "accuracy": 0.2134,
        "log_loss": 0.9107,
        "brier_score": 0.2090,
        "ece_global": 0.1800,
    },
}
times = {
    "MAP": 3.94,
    "Laplace Approx": 18.90,
    "MC Dropout": 186.93,
    "SGHMC": 186.31,
    # "Deep Ensemble": 42.43,
    "Temp Scaling": 7.33
}

# ------------------------
# Now plot!
plot_inference_results(metrics, times)
plot_id_vs_ood_grouped(metrics, ood_metrics)
