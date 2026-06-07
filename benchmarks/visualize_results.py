import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 1. load the data
with open("benchmarks/matrix_raw_results.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 2. clean the data (drop the pytorch warmup outlier for a fair ttft average)
df_clean = df.drop(index=0)

sns.set_theme(style="darkgrid", palette="pastel")

# helper function to save plots cleanly
def save_plot(fig, filename):
    plt.tight_layout()
    fig.savefig(f"benchmarks/{filename}", dpi=350, bbox_inches='tight')
    print(f"saved: benchmarks/{filename}")

print("generating benchmark visualizations...")

# PLOT 1: Time To First Token (TTFT)
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.barplot(
    data=df_clean, x="k_chunks", y="ttft_ms", hue="embedding_device", 
    capsize=.1, err_kws={'linewidth': 1.5}, ax=ax1
)
ax1.set_title("Time-To-First-Token (TTFT) by Configuration", fontsize=14, pad=15)
ax1.set_ylabel("TTFT (milliseconds)")
ax1.set_xlabel("Retrieval Volume (Top-K Chunks)")
save_plot(fig1, "chart_ttft.png")

# PLOT 2: Inter-Token Latency (ITL)
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(
    data=df, x="k_chunks", y="itl_ms", hue="embedding_device", 
    capsize=.1, err_kws={'linewidth': 1.5}, ax=ax2
)
ax2.set_title("Inter-Token Latency (Compute Thrashing)", fontsize=14, pad=15)
ax2.set_ylabel("ITL (ms / token)")
ax2.set_xlabel("Retrieval Volume (Top-K Chunks)")
save_plot(fig2, "chart_itl.png")

# PLOT 3: Peak VRAM Consumption
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.barplot(
    data=df, x="k_chunks", y="peak_vram_gb", hue="embedding_device", 
    capsize=.1, err_kws={'linewidth': 1.5}, ax=ax3
)
ax3.set_title("Peak VRAM Footprint (KV-Cache Expansion)", fontsize=14, pad=15)
ax3.set_ylabel("Peak VRAM (GB)")
ax3.set_xlabel("Retrieval Volume (Top-K Chunks)")
# add a red line showing 4050 hardware limit (6gb)
ax3.axhline(6.0, color='red', linestyle='--', alpha=0.5, label='RTX 4050 Limit (6GB)')
ax3.legend()
save_plot(fig3, "chart_vram.png")

print("done! saved to benchmarks folder.")