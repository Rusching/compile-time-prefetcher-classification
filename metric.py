import os
import re
import pandas as pd

# 定义关键指标的正则表达式
metrics = {
    "Core_0_instructions": r"Core_0_instructions\s+(\d+)",
    "Core_0_cycles": r"Core_0_cycles\s+(\d+)",
    "Core_0_IPC": r"Core_0_IPC\s+([\d\.]+)",
    "Core_0_branch_prediction_accuracy": r"Core_0_branch_prediction_accuracy\s+([\d\.]+)",
    "Core_0_branch_MPKI": r"Core_0_branch_MPKI\s+([\d\.]+)",
    "Core_0_L1D_total_access": r"Core_0_L1D_total_access\s+(\d+)",
    "Core_0_L1D_total_hit": r"Core_0_L1D_total_hit\s+(\d+)",
    "Core_0_L1D_total_miss": r"Core_0_L1D_total_miss\s+(\d+)",
    "Core_0_L1D_loads": r"Core_0_L1D_loads\s+(\d+)",
    "Core_0_L1D_load_hit": r"Core_0_L1D_load_hit\s+(\d+)",
    "Core_0_L1D_load_miss": r"Core_0_L1D_load_miss\s+(\d+)",
    "Core_0_L1D_RFOs": r"Core_0_L1D_RFOs\s+(\d+)",
    "Core_0_L1D_RFO_hit": r"Core_0_L1D_RFO_hit\s+(\d+)",
    "Core_0_L1D_RFO_miss": r"Core_0_L1D_RFO_miss\s+(\d+)",
    "Core_0_L1D_average_miss_latency": r"Core_0_L1D_average_miss_latency\s+([\d\.]+)"
}

# 定义文件夹路径
folder_path = "./exp1c"  # 请根据实际路径调整

# 初始化结果列表
results = []

# 遍历文件夹中的 .out 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".out"):
        trace_name, prefetcher_name = file_name.split("_")[:2]
        file_path = os.path.join(folder_path, file_name)
        
        # 读取文件内容
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()
        
        # 提取指标
        data = {"Trace": trace_name, "Prefetcher": prefetcher_name}
        for metric, regex in metrics.items():
            match = re.search(regex, content)
            if match:
                data[metric] = float(match.group(1)) if "." in match.group(1) else int(match.group(1))
            else:
                data[metric] = None  # 如果没有匹配到，填入 None
        
        # 计算派生指标
        if data["Core_0_L1D_total_access"]:
            data["L1D_Hit_Rate"] = data["Core_0_L1D_total_hit"] / data["Core_0_L1D_total_access"] * 100
            data["L1D_Miss_Rate"] = data["Core_0_L1D_total_miss"] / data["Core_0_L1D_total_access"] * 100
        else:
            data["L1D_Hit_Rate"] = None
            data["L1D_Miss_Rate"] = None
        
        if data["Core_0_L1D_loads"]:
            data["L1D_Load_Hit_Rate"] = data["Core_0_L1D_load_hit"] / data["Core_0_L1D_loads"] * 100
            data["L1D_Load_Miss_Rate"] = data["Core_0_L1D_load_miss"] / data["Core_0_L1D_loads"] * 100
        else:
            data["L1D_Load_Hit_Rate"] = None
            data["L1D_Load_Miss_Rate"] = None
        
        if data["Core_0_L1D_RFOs"]:
            data["L1D_RFO_Hit_Rate"] = data["Core_0_L1D_RFO_hit"] / data["Core_0_L1D_RFOs"] * 100
            data["L1D_RFO_Miss_Rate"] = data["Core_0_L1D_RFO_miss"] / data["Core_0_L1D_RFOs"] * 100
        else:
            data["L1D_RFO_Hit_Rate"] = None
            data["L1D_RFO_Miss_Rate"] = None
        
        # 添加到结果列表
        results.append(data)

# 转换为 Pandas DataFrame
df = pd.DataFrame(results)

# 保存为 CSV 文件
output_file = "prefetcher_analysis_results.csv"
df.to_csv(output_file, index=False)

print(f"数据已处理并保存到 {output_file}")
