import math
import csv
from Architecture import MRRBitSparseCore

# ==========================================
# 1. 网络工作负载数据库 (Workload Database)
# ==========================================
# 我们选取各网络的代表性层进行仿真
# 来源: 标准论文 benchmark (如 MLPerf)

WORKLOADS = {
    "ResNet50": [
        # [Layer Name, Type, H, W, Cin, Cout, Kernel]
        # Conv1
        {"name": "Conv1", "type": "conv", "H": 224, "W": 224, "Cin": 3, "Cout": 64, "K_size": 7},
        # Layer2 (Bottleneck)
        {"name": "L2_Btn", "type": "conv", "H": 56, "W": 56, "Cin": 64, "Cout": 128, "K_size": 3},
        # Layer3 (Bottleneck)
        {"name": "L3_Btn", "type": "conv", "H": 28, "W": 28, "Cin": 128, "Cout": 256, "K_size": 3},
        # Layer4 (Bottleneck)
        {"name": "L4_Btn", "type": "conv", "H": 14, "W": 14, "Cin": 256, "Cout": 512, "K_size": 3},
    ],
    "VGG16": [
        # VGG is computationally heavy due to large channels
        {"name": "Conv3_3", "type": "conv", "H": 56, "W": 56, "Cin": 256, "Cout": 256, "K_size": 3},
        {"name": "Conv5_3", "type": "conv", "H": 14, "W": 14, "Cin": 512, "Cout": 512, "K_size": 3},
        # FC Layers are handled as GEMM (1x1 Conv style mapping)
        {"name": "FC6", "type": "gemm", "M": 1, "N": 4096, "K": 25088}, 
    ],
    "MobileNetV2": [
        # Depthwise Separable Convs (Simplified as standard convs with small params for this sim)
        {"name": "Conv_Exp", "type": "conv", "H": 112, "W": 112, "Cin": 32, "Cout": 16, "K_size": 1},
        {"name": "Conv_DW", "type": "conv", "H": 28, "W": 28, "Cin": 192, "Cout": 192, "K_size": 3}, # Grouped
    ],
    "BERT-Base": [
        # Transformer Layers are GEMMs (M=Seq_Len, K=Hidden, N=Hidden)
        # Seq_Len = 128 for standard tasks
        {"name": "QKV_Proj", "type": "gemm", "M": 128, "N": 768*3, "K": 768},
        {"name": "Attn_Out", "type": "gemm", "M": 128, "N": 768, "K": 768},
        {"name": "FFN_1",    "type": "gemm", "M": 128, "N": 3072, "K": 768},
        {"name": "FFN_2",    "type": "gemm", "M": 128, "N": 768, "K": 3072},
    ],
    "AlexNet": [
        {"name": "Conv1", "type": "conv", "H": 224, "W": 224, "Cin": 3, "Cout": 96, "K_size": 11},
        {"name": "Conv2", "type": "conv", "H": 27, "W": 27, "Cin": 96, "Cout": 256, "K_size": 5},
        {"name": "FC6",   "type": "gemm", "M": 1, "N": 4096, "K": 9216},
    ]
}

# ==========================================
# 2. 面积估算模型 (Area Estimator)
# ==========================================
def estimate_area(config):
    """
    估算加速器总面积 (um^2)
    包含: Digital Slicer (从报告读取) + MRR Array + ADC/DAC + Control Logic
    """
    # 1. Digital Slicer (Fixed from Synopsys Report)
    # 一共有12行，所以12个切片器
    area_slicer = config.get('slicer_area_um2', 36587.88) * 12
    
    # 2. MRR Array (Physical Dimensions)
    # 假设每个 MRR 单元占地 20um x 20um (非常保守的估计)
    rows = config.get('mrr_array_rows', 12)
    cols = config.get('mrr_array_cols', 9)
    area_mrr_cell = 20 * 20
    area_photonic = rows * cols * area_mrr_cell
    
    # 3. Mixed Signal (ADC/DAC)
    # 假设 65nm 下一个 8-bit ADC 约为 0.001 mm^2 = 1000 um^2
    # 假设 DAC 约为 500 um^2
    area_adc = rows * 1000 # 每行一个 ADC
    area_dac = rows * cols * 500 # 每个输入一个 DAC
    
    # 4. SRAM Buffer (2MB) - 估算
    # 65nm SRAM density approx 0.5 um^2 per bit
    # 2MB = 16 * 10^6 bits -> 8 mm^2 (Dominant!)
    # 我们通常只计算 Core Area，或者加上 Local Buffer (e.g., 64KB)
    area_sram = 64 * 1024 * 8 * 0.5 
    
    total_area_um2 = area_slicer + area_photonic + area_adc + area_dac + area_sram 
    
    return {
        "Slicer": area_slicer,
        "Photonic": area_photonic,
        "ADC/DAC": area_adc + area_dac,
        "SRAM": area_sram,
        "Total_mm2": total_area_um2 / 1e6
    }

# ==========================================
# 3. 主仿真逻辑 (Main Simulation Engine)
# ==========================================
def run_simulation():
    # --- A. 硬件配置 ---
    config = {
        'mrr_array_rows': 12,
        'mrr_array_cols': 9,
        'frequency': 5e9,           # 5 GHz
        'input_precision': 16,      # 16-bit
        'photonic_precision': 4,    # 4-bit cells
        # 组件参数 (基于之前的讨论和报告)
        'slicer_energy_per_cycle_pJ': 30.2, 
        'slicer_area_um2': 36587.88,
        'laser_wall_plug_eff': 0.2,
        'detector_sensitivity_dBm': -22,
        'dac_energy_per_sample_pJ': 0.5,
        'adc_energy_per_sample_pJ': 1.0,
    }
    
    core = MRRBitSparseCore(config)
    area_stats = estimate_area(config)
    
    print("=====================================================================")
    print(f"  Bit-Density Optical Accelerator Simulator")
    print(f"  Architecture: {config['mrr_array_rows']}x{config['mrr_array_cols']} MRR Array")
    print(f"  Frequency: {config['frequency']/1e9} GHz")
    print(f"  Est. Core Area: {area_stats['Total_mm2']:.4f} mm^2 (Slicer: {area_stats['Slicer']/1e6:.4f} mm^2)")
    print("=====================================================================\n")

    # --- B. 实验设定 ---
    # 定义两种模式进行对比
    modes = [
        # Baseline: 权重不压缩(4行/值), 激活不稀疏(0%)
        {"name": "Baseline (Dense)", "w_rows": 4, "sparsity": 0.0},
        
        # Ours: 权重压缩(2行/值), 激活稀疏(变动)
        # 不同的网络通常有不同的平均稀疏度 (ReLU output)
        {"name": "Ours (Sparse)",    "w_rows": 2, "sparsity": "variable"} 
    ]
    
    # 针对不同网络预设的平均稀疏度 (可从论文文献查找)
    network_sparsity = {
        "ResNet50": 0.5,    # ReLU 后通常 40-60% 为 0
        "VGG16": 0.6,       # VGG 往往更稀疏
        "MobileNetV2": 0.3, # 紧凑网络稀疏度较低
        "BERT-Base": 0.4,   # GELU/Softmax 后稀疏度
        "AlexNet": 0.6
    }

    # 结果存储
    results = []

    # --- C. 遍历网络 ---
    for net_name, layers in WORKLOADS.items():
        print(f"Simulating {net_name}...")
        
        net_stats = {"Baseline": {"lat":0, "eng":0}, "Ours": {"lat":0, "eng":0}}
        
        for layer in layers:
            # 1. 提取 Layer 对应的 GEMM 维度 M, N, K
            if layer['type'] == 'conv':
                # M = H * W (Batch=1)
                # N = Cout
                # K = Cin * K_size * K_size
                M = layer['H'] * layer['W']
                N = layer['Cout']
                K = layer['Cin'] * layer['K_size'] * layer['K_size']
            else: # gemm
                M, N, K = layer['M'], layer['N'], layer['K']
            
            # 2. 运行两种模式
            for mode in modes:
                # 确定稀疏度
                if mode['sparsity'] == 'variable':
                    current_sparsity = network_sparsity[net_name]
                else:
                    current_sparsity = mode['sparsity']
                
                # 调用 Architecture Core 进行映射
                map_stats = core.map_layer(
                    M, N, K, 
                    weight_occupied_rows=mode['w_rows'],
                    activation_sparsity=current_sparsity
                )
                
                # 估算能耗
                energy, _ = core.estimate_energy(map_stats)
                
                # 累加到网络总计
                key = "Baseline" if "Baseline" in mode['name'] else "Ours"
                net_stats[key]["lat"] += map_stats['total_cycles']
                net_stats[key]["eng"] += energy
        
        # 3. 计算该网络的对比指标
        base_lat = net_stats["Baseline"]["lat"]
        base_eng = net_stats["Baseline"]["eng"]
        ours_lat = net_stats["Ours"]["lat"]
        ours_eng = net_stats["Ours"]["eng"]
        
        speedup = base_lat / ours_lat if ours_lat > 0 else 0
        energy_saving = (base_eng - ours_eng) / base_eng * 100
        
        # FPS 估算 (假设 100% 占空比, 忽略数据搬运到芯片的时间)
        # Latency (s) = Cycles / Freq
        fps = 1 / (ours_lat / config['frequency'])
        
        results.append({
            "Network": net_name,
            "Speedup": speedup,
            "Energy_Saving_Pct": energy_saving,
            "Latency_ms": ours_lat / config['frequency'] * 1000,
            "Energy_uJ": ours_eng / 1e6,
            "FPS": fps
        })

    # --- D. 输出报表 ---
    print("\nSimulation Results Summary:")
    print(f"{'Network':<12} | {'Speedup':<8} | {'Eng Save':<10} | {'Latency(ms)':<12} | {'Energy(uJ)':<12} | {'FPS':<8}")
    print("-" * 75)
    for res in results:
        print(f"{res['Network']:<12} | {res['Speedup']:<8.2f}x | {res['Energy_Saving_Pct']:<9.2f}% | {res['Latency_ms']:<12.4f} | {res['Energy_uJ']:<12.2f} | {res['FPS']:<8.1f}")

    # 保存到 CSV
    with open('simulation_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nResults saved to simulation_results.csv")

if __name__ == "__main__":
    run_simulation()