import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '../prefetcher_analysis_results.csv'
data = pd.read_csv(file_path)

# clean data
data['Trace'] = data['Trace'].str.strip()  # clean white spaces
pivot_hit_rate = data.pivot(index='Trace', columns='Prefetcher', values='L1D_Hit_Rate')

# set graph style
sns.set_theme(style="whitegrid")
colors = sns.color_palette("Set2", len(pivot_hit_rate.columns))

# draw hit rate graph
pivot_hit_rate.plot(kind='bar', figsize=(12, 6), color=colors, edgecolor='black')
plt.title("L1D Cache Hit Rate by Trace and Prefetcher", fontsize=16)
plt.ylabel("L1D Cache Hit Rate (%)", fontsize=12)
plt.xlabel("Trace", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="Prefetcher", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("L1D_Cache_Hit_Rate.png")
plt.show()

# draw IPC graph
pivot_ipc = data.pivot(index='Trace', columns='Prefetcher', values='Core_0_IPC')
pivot_ipc.plot(kind='bar', figsize=(12, 6), color=colors, edgecolor='black')
plt.title("Core IPC by Trace and Prefetcher", fontsize=16)
plt.ylabel("Core IPC", fontsize=12)
plt.xlabel("Trace", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="Prefetcher", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("Core_IPC.png")
plt.show()

# draw Miss Latency graph
pivot_latency = data.pivot(index='Trace', columns='Prefetcher', values='Core_0_L1D_average_miss_latency')
pivot_latency.plot(kind='bar', figsize=(12, 6), color=colors, edgecolor='black')
plt.title("L1D Average Miss Latency by Trace and Prefetcher", fontsize=16)
plt.ylabel("L1D Miss Latency (cycles)", fontsize=12)
plt.xlabel("Trace", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="Prefetcher", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("L1D_Average_Miss_Latency.png")
plt.show()
