import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --- 1. Training loss curve (multi-task joint loss, from real_results_log.txt) ---
epochs = [1, 20, 40, 60, 80, 100, 120, 140]
joint_loss = [2.5205, 2.2377, 2.1727, 2.1347, 2.1158, 2.1006, 2.0952, 2.0760]

plt.figure(figsize=(6, 4))
plt.plot(epochs, joint_loss, marker="o", color="#2b6cb0", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Joint loss  (0.5·MSE_cost + 1.5·BCE_sched + 1.5·BCE_risk)")
plt.title("Multi-task network training loss\n(real DoD FPDS-NG construction data)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "loss_curve.png"), dpi=150)
plt.close()

#  Results heatmap: model x task, using the metric reported as primary
#     for each task in the paper (R2 for cost, AUC for schedule/risk) 
models = ["Multi-Task Net", "Single-Task RF", "Single-Task MLP"]
tasks = ["Cost growth (R2)", "Schedule overrun (AUC)", "Overall risk (AUC)"]

# rows = models, cols = tasks (from real_results_log.txt final RESULTS table)
data = np.array([
    [0.0342, 0.6523, 0.6542],   # Multi-Task Net
    [-0.0151, 0.7073, 0.7015],  # Single-Task RF
    [-0.1127, 0.6480, 0.6506],  # Single-Task MLP
])

fig, ax = plt.subplots(figsize=(6.5, 4))
im = ax.imshow(data, cmap="RdYlGn", vmin=-0.15, vmax=0.75, aspect="auto")

ax.set_xticks(range(len(tasks)))
ax.set_xticklabels(tasks, rotation=15, ha="right")
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models)

for i in range(len(models)):
    for j in range(len(tasks)):
        ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                 color="black", fontsize=11, fontweight="bold")

ax.set_title("Test-set performance by model and task\n(train FY2009-14, test FY2015-17, real data)")
fig.colorbar(im, ax=ax, shrink=0.8, label="Score (R2 or AUC)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "results_heatmap.png"), dpi=150)
plt.close()

print("Wrote figures/loss_curve.png and figures/results_heatmap.png")
