# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import time

# # # ==========================================
# # # 1. 核心编码算法: Bit-Density (MSB 优先)
# # # ==========================================
# # def encode_msb_priority(group, max_k=8, mrr_cap=4):
# #     """
# #     基于 MSB 优先（大 S 优先）的致密编码逻辑
    
# #     参数:
# #     - group: 输入的权重分组 (int向量)
# #     - max_k: 最大允许切片数
# #     - mrr_cap: MRR 最大能级 (L_max = mrr_cap + 1)
# #       例如 mrr_cap=4, 代表能级 {0, 1, 2, 3, 4}, 共5个电平
    
# #     返回:
# #     - k: 实际使用的切片数
# #     - slices_s: 每个切片的位移 S
# #     - slices_l: 每个切片的能级 L
# #     """
# #     # 预处理：取绝对值（符号位单独处理）
# #     original = group.astype(np.float64)
# #     target = np.abs(original).astype(np.int64)
    
# #     if np.sum(target) == 0:
# #         return 0, [], []

# #     residues = target.copy()
# #     slices_s = []
# #     slices_l = []
    
# #     # 计算硬件有效位宽 (用于确定搜索窗口大小)
# #     # 例如 Cap=4 -> log2(4)=2 -> 2+1=3 bit window
# #     hardware_bit_width = int(np.floor(np.log2(mrr_cap))) + 1

# #     for k in range(1, max_k + 1):
# #         max_val = np.max(residues)
# #         if max_val == 0: 
# #             break
            
# #         # 1. 锚定最高有效位 (MSB)
# #         j_msb = int(np.floor(np.log2(max_val)))
        
# #         # 2. 定义 S 的搜索窗口: [MSB - bit_width + 1, MSB]
# #         search_start = max(0, j_msb - hardware_bit_width + 1)
# #         search_end = j_msb 
        
# #         best_s = -1
# #         min_cost = float('inf')
# #         best_levels = None
        
# #         # 3. 策略竞争：在窗口内寻找最佳 S
# #         for s_candidate in range(search_start, search_end + 1):
# #             # 计算当前 S 下的量化能级
# #             levels = residues >> s_candidate
# #             levels = np.clip(levels, 0, mrr_cap) # 硬件截断
            
# #             # 计算重构成分与残差
# #             current_comp = levels << s_candidate
# #             temp_residue = residues - current_comp
            
# #             # --- 代价函数设计 (核心) ---
# #             # 主要目标: 最小化残差能量 (L1范数)
# #             energy_cost = np.sum(temp_residue) 
            
# #             # 辅助目标: MSB 优先 (负惩罚 = 奖励大 S)
# #             # 这强迫算法在残差相近时，优先切掉高位，保持残差结构的规整性
# #             penalty = -s_candidate * 0.1 
            
# #             total_cost = energy_cost + penalty
            
# #             if total_cost < min_cost:
# #                 min_cost = total_cost
# #                 best_s = s_candidate
# #                 best_levels = levels

# #         # 兜底保护
# #         if best_levels is None:
# #              best_s = 0
# #              best_levels = np.clip(residues, 0, mrr_cap)

# #         # 4. 执行切片
# #         slices_s.append(best_s)
# #         slices_l.append(best_levels)
        
# #         # 更新残差
# #         current_components = best_levels << best_s
# #         residues -= current_components
        
# #     return len(slices_s), slices_s, slices_l

# # # ==========================================
# # # 2. 物理层仿真: 噪声注入与重构
# # # ==========================================
# # def simulate_physical_recon(original, slices_s, slices_l, mrr_cap, noise_sigma=0.01):
# #     """
# #     模拟光计算物理过程中的噪声影响
    
# #     参数:
# #     - noise_sigma: 归一化噪声标准差 (相对于满量程 1.0)
# #       0.01 代表 1% 的误差 (典型 MRR 热噪声/驱动误差)
# #     """
# #     target = np.abs(original)
    
# #     # 初始化物理重构值
# #     recon_val = np.zeros_like(target, dtype=np.float64)
    
# #     for s, l_ideal in zip(slices_s, slices_l):
# #         # A. DAC 转换: 数字 L -> 模拟透射率 T (归一化 [0, 1])
# #         # L_ideal 是 0 到 mrr_cap 的整数
# #         t_ideal = l_ideal.astype(np.float64) / mrr_cap
        
# #         # B. 物理信道: 注入高斯白噪声
# #         # 噪声幅度是相对于满量程的
# #         noise = np.random.normal(0, noise_sigma, size=t_ideal.shape)
# #         t_noisy = t_ideal + noise
        
# #         # C. 物理检测: T -> 模拟电流 -> 数字值
# #         # 还原到数值域
# #         l_noisy = t_noisy * mrr_cap
        
# #         # 累加 (移位由电域移位器完成，假设它是数字的，无误差，或误差已包含在整体中)
# #         recon_val += l_noisy * (2.0 ** s)
        
# #     # 计算 SQNR
# #     err = target - recon_val
# #     sig_pow = np.sum(target ** 2)
# #     err_pow = np.sum(err ** 2)
    
# #     if err_pow == 0:
# #         return 99.9
    
# #     sqnr = 10 * np.log10(sig_pow / err_pow)
# #     return sqnr

# # # ==========================================
# # # 3. 主实验程序
# # # ==========================================
# # def run_experiment():
# #     print("--- 开始 Bit-Density 本地仿真实验 ---")
    
# #     # 1. 生成仿真数据 (模拟一层典型的 CNN 权重)
# #     np.random.seed(42)
# #     data_size = 4096
# #     print(f"生成随机权重数据 (Size={data_size})...")
# #     # 正态分布，模拟权重的钟形曲线
# #     weights_raw = np.random.randn(data_size)
# #     # 量化到 INT16 范围 [0, 32767]
# #     scale = 32767.0 / np.max(np.abs(weights_raw))
# #     weights_int = np.round(np.abs(weights_raw) * scale).astype(np.int32)
    
# #     # 分组 (WDM size = 9, 模拟一个波导上的计算组)
# #     # 丢弃末尾不足一组的数据
# #     num_groups = len(weights_int) // 9
# #     groups = weights_int[:num_groups*9].reshape(-1, 9)
    
# #     # 2. 定义实验配置
# #     # (MRR_Cap, 标签)
# #     caps_config = [
# #         (1, "1-bit (L=2)"),
# #         (2, "L=3"),
# #         (3, "2-bit (L=4)"),
# #         (4, "L=5 [Ours]"),   # 重点关注
# #         (5, "L=6"),
# #         (6, "L=7"),
# #         (7, "3-bit (L=8)"),
# #         (15, "4-bit (L=16)")
# #     ]

# #     # 噪声水平 (相对于满量程 1.0)
# #     # 0.005 (0.5%), 0.01 (1.0%), 0.015 (1.5%)
# #     noise_levels = [0.005, 0.010, 0.015] 
    
# #     results = []

# #     print(f"正在扫描 {len(caps_config)} 种能级配置，每种测试 {len(noise_levels)} 个噪声水平...")
# #     start_time = time.time()

# #     for mrr_cap, label in caps_config:
# #         # --- 步骤 A: 算法编码 ---
# #         # 这一步是确定的，与噪声无关，计算 K 值
# #         ks = []
# #         all_slices_info = [] 
        
# #         for g in groups:
# #             k, s_list, l_list = encode_msb_priority(g, max_k=8, mrr_cap=mrr_cap)
# #             ks.append(k)
# #             all_slices_info.append((s_list, l_list))
            
# #         avg_k = np.mean(ks)
# #         saving = (8.0 - avg_k) / 8.0 * 100
        
# #         # --- 步骤 B: 物理加噪评估 ---
# #         for noise in noise_levels:
# #             sqnrs = []
# #             for i, g in enumerate(groups):
# #                 s_list, l_list = all_slices_info[i]
# #                 sqnr = simulate_physical_recon(g, s_list, l_list, mrr_cap, noise_sigma=noise)
# #                 sqnrs.append(sqnr)
                
# #             avg_sqnr = np.mean(sqnrs)
            
# #             results.append({
# #                 "Cap": mrr_cap,
# #                 "L_max": mrr_cap + 1,
# #                 "Label": label,
# #                 "Avg_K": avg_k,
# #                 "Noise_Sigma": noise,
# #                 "Noise_Label": f"Noise={noise*100:.1f}%",
# #                 "Avg_SQNR": avg_sqnr,
# #                 "Saving_Percent": saving
# #             })
            
# #         print(f"  -> 完成 {label}: Avg_K={avg_k:.2f}, Saving={saving:.1f}%")

# #     print(f"仿真结束，耗时 {time.time() - start_time:.2f} 秒。")
    
# #     # ==========================================
# #     # 4. 数据可视化
# #     # ==========================================
# #     df = pd.DataFrame(results)
    
# #     # 设置绘图风格
# #     sns.set_theme(style="whitegrid")
# #     fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
# #     # 图 1: 算法收益 (Saving vs Cap)
# #     # 只需要取一个噪声水平的数据来画 K
# #     df_k = df[df['Noise_Sigma'] == noise_levels[0]]
    
# #     sns.barplot(ax=axes[0], data=df_k, x='Label', y='Saving_Percent', palette='viridis', edgecolor='black')
# #     axes[0].set_title('Algorithm Benefit: Bit-Sparsity Saving (%)', fontsize=14, fontweight='bold')
# #     axes[0].set_ylabel('Saving vs. 8-bit Serial (%)', fontsize=12)
# #     axes[0].set_xlabel('MRR Capability (L_max)', fontsize=12)
# #     axes[0].tick_params(axis='x', rotation=45)
    
# #     # 在柱子上标注数值
# #     for i, row in df_k.reset_index().iterrows():
# #         axes[0].text(i, row['Saving_Percent'] + 1, f"{row['Saving_Percent']:.1f}%", 
# #                      color='black', ha="center", fontweight='bold')

# #     # 图 2: 物理代价 (SQNR vs Cap)
# #     sns.lineplot(ax=axes[1], data=df, x='Label', y='Avg_SQNR', hue='Noise_Label', 
# #                  style='Noise_Label', markers=True, dashes=False, linewidth=2.5, markersize=9)
    
# #     axes[1].set_title('Physical Cost: Reconstruction Robustness (SQNR)', fontsize=14, fontweight='bold')
# #     axes[1].set_ylabel('Effective SQNR (dB)', fontsize=12)
# #     axes[1].set_xlabel('MRR Capability (L_max)', fontsize=12)
# #     axes[1].tick_params(axis='x', rotation=45)
# #     axes[1].grid(True, linestyle='--', alpha=0.7)
    
# #     # 标注 L=5 的“甜点”位置
# #     # 找到 L=5 对应的数据点
# #     l5_data = df[df['Label'].str.contains("L=5")]
# #     if not l5_data.empty:
# #         # 取中间噪声水平的点
# #         mid_noise = noise_levels[1]
# #         val = l5_data[l5_data['Noise_Sigma'] == mid_noise]['Avg_SQNR'].values[0]
# #         # 获取 x 轴坐标 (L=5 是列表中的第4个，索引为3)
# #         axes[1].annotate('Sweet Spot (L=5)\nHigh Robustness', xy=(3, val), xytext=(3, val+5),
# #                          arrowprops=dict(facecolor='red', shrink=0.05),
# #                          fontsize=11, color='red', fontweight='bold', ha='center')

# #     plt.tight_layout()
# #     plt.savefig('output/bit_density_simulation_results.png', dpi=300)
# #     print("\n结果图表已保存为 'output/bit_density_simulation_results.png'")
    
# #     # 打印 L=4 vs L=5 的关键对比数据
# #     print("\n=== 关键对比: L=4 (Standard) vs L=5 (Ours) ===")
# #     cols = ['Label', 'Noise_Label', 'Avg_K', 'Avg_SQNR']
# #     subset = df[df['Label'].str.contains("L=4|L=5")].sort_values(by=['Label', 'Noise_Sigma'])
# #     print(subset[cols].to_string(index=False))

# # if __name__ == "__main__":
# #     run_experiment()
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm

# # ==========================================
# # 1. 核心算法: 物理感知的位致密编码 (Bit-Density)
# # ==========================================
# def encode_msb_priority(group, max_k=8, mrr_cap=4):
#     """
#     针对 MRR 物理特性优化的编码算法
#     - MSB Priority: 优先使用大步进 S 清除高位能量
#     - mrr_cap: 硬件最大能级 (L_max = mrr_cap + 1)
#     """
#     # 预处理：转为绝对值进行编码，符号位单独处理
#     original = group.astype(np.float64)
#     target = np.abs(original).astype(np.int64)
    
#     if np.sum(target) == 0:
#         return 0, [], []

#     residues = target.copy()
#     slices_s = [] # 记录每一步的移位量 S
#     slices_l = [] # 记录每一步的能级量 L
    
#     # 硬件位宽 (用于确定搜索窗口)
#     # 例如 Cap=4 (0-4), bit_width=3 (能覆盖 100)
#     # 例如 Cap=3 (0-3), bit_width=2 (能覆盖 11)
#     if mrr_cap == 0: return 0, [], []
#     hardware_bit_width = int(np.floor(np.log2(mrr_cap))) + 1

#     for k in range(1, max_k + 1):
#         max_val = np.max(residues)
#         if max_val == 0: 
#             break
            
#         j_msb = int(np.floor(np.log2(max_val)))
        
#         # 动态搜索窗口: 寻找最佳的 S
#         # 策略: 允许 S 向下探测，利用非标准能级拟合残差
#         search_start = max(0, j_msb - hardware_bit_width + 1)
#         search_end = j_msb 
        
#         best_s = -1
#         min_cost = float('inf')
#         best_levels = None
        
#         for s_candidate in range(search_start, search_end + 1):
#             # 1. 尝试量化
#             levels = residues >> s_candidate
#             levels = np.clip(levels, 0, mrr_cap)
            
#             # 2. 计算残差
#             current_comp = levels << s_candidate
#             temp_residue = residues - current_comp
            
#             # 3. 代价函数 (核心)
#             # Energy Cost: 剩余残差的 L1 范数
#             energy_cost = np.sum(temp_residue) 
#             # Penalty: 负惩罚 = 奖励大 S。强制算法优先切高位，保证收敛速度。
#             penalty = -s_candidate * 0.1 
            
#             total_cost = energy_cost + penalty
            
#             if total_cost < min_cost:
#                 min_cost = total_cost
#                 best_s = s_candidate
#                 best_levels = levels

#         # 兜底机制
#         if best_levels is None:
#              best_s = 0
#              best_levels = np.clip(residues, 0, mrr_cap)

#         slices_s.append(best_s)
#         slices_l.append(best_levels)
        
#         # 更新残差
#         current_components = best_levels << best_s
#         residues -= current_components
        
#     return len(slices_s), slices_s, slices_l

# # ==========================================
# # 2. 物理层仿真: 噪声注入与重构
# # ==========================================
# def simulate_physical_recon(original, slices_s, slices_l, mrr_cap, noise_sigma=0.01):
#     """
#     模拟光计算物理链路
#     - noise_sigma: 归一化后的噪声标准差 (相对于满量程 1.0)
#     """
#     target = np.abs(original)
#     recon_val = np.zeros_like(target, dtype=np.float64)
    
#     # 模拟 DAC -> MRR -> PD 链路
#     for s, l_ideal in zip(slices_s, slices_l):
#         # A. 理想控制信号 (归一化到 0-1)
#         t_ideal = l_ideal.astype(np.float64) / mrr_cap
        
#         # B. 注入物理噪声 (热噪声 + 驱动误差)
#         noise = np.random.normal(0, noise_sigma, size=t_ideal.shape)
#         t_noisy = t_ideal + noise
        
#         # C. 物理还原 (光强 -> 电流 -> 数值)
#         # 假设线性探测，且光强非负 (简单模型 clip 0)
#         # t_noisy = np.clip(t_noisy, 0, 1.5) # 允许一定过冲，但不允许负光
#         l_noisy = t_noisy * mrr_cap
        
#         recon_val += l_noisy * (2.0 ** s)
        
#     # 计算 SQNR
#     err = target - recon_val
#     sig_pow = np.sum(target ** 2)
#     err_pow = np.sum(err ** 2)
    
#     if err_pow == 0: return 99.9
#     return 10 * np.log10(sig_pow / err_pow)

# # ==========================================
# # 3. 数据加载器: 提取真实模型权重
# # ==========================================
# def load_quantized_weights(model_name='resnet18', wdm_size=9, limit_samples=5000):
#     print(f"--- Loading {model_name} from torchvision ---")
#     if model_name == 'resnet18':
#         model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#     elif model_name == 'vgg16':
#         model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
#     else:
#         model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    
#     all_groups = []
#     print("Extracting Convolutional Layers...")
    
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d):
#             # 获取权重并展平
#             w = module.weight.detach().cpu().numpy().flatten()
            
#             # INT16 量化模拟
#             w_max = np.max(np.abs(w))
#             if w_max == 0: continue
#             scale = 32767.0 / w_max
#             w_int = np.round(w * scale).astype(np.int32)
            
#             # Padding 以适配 WDM 分组
#             remainder = len(w_int) % wdm_size
#             if remainder != 0:
#                 w_int = np.pad(w_int, (0, wdm_size - remainder), 'constant')
            
#             groups = w_int.reshape(-1, wdm_size)
            
#             # 随机采样一部分数据，避免仿真时间过长
#             if len(groups) > 0:
#                 # 简单打乱
#                 idx = np.random.permutation(len(groups))
#                 # 每层取一部分，保证多样性
#                 sample_size = min(len(groups), 200) 
#                 all_groups.append(groups[idx[:sample_size]])
    
#     final_data = np.vstack(all_groups)
    
#     # 如果总数超过限制，再截断
#     if len(final_data) > limit_samples:
#         final_data = final_data[:limit_samples]
        
#     print(f"Total grouped vectors for simulation: {len(final_data)}")
#     return final_data

# # ==========================================
# # 4. 主实验流程
# # ==========================================
# if __name__ == "__main__":
#     # 1. 准备数据
#     # WDM_SIZE = 9 (适配 3x3 卷积展开)
#     real_weights = load_quantized_weights('resnet18', wdm_size=9)
    
#     # 2. 配置实验参数
#     # Cap 对应 L_max = Cap + 1
#     # Cap 1 = 1-bit (L=2)
#     # Cap 3 = 2-bit (L=4)
#     # Cap 4 = L=5 (Ours)
#     # Cap 7 = 3-bit (L=8)
#     configs = [
#         (1, "1-bit (L=2)"),
#         (2, "Non-Std (L=3)"),
#         (3, "2-bit (L=4)"),
#         (4, "Ours (L=5)"), 
#         (5, "Non-Std (L=6)"),
#         (6, "Non-Std (L=7)"),
#         (7, "3-bit (L=8)"),
#         (15, "4-bit (L=16)")
#     ]
    
#     noise_sigma = 0.01 # 1% 噪声水平
#     results = []
    
#     print(f"\n--- Starting Simulation (Noise = {noise_sigma*100}%) ---")
    
#     for cap, label in tqdm(configs):
#         k_list = []
#         sqnr_list = []
        
#         for group in real_weights:
#             # Encoding
#             k, s_list, l_list = encode_msb_priority(group, max_k=8, mrr_cap=cap)
            
#             # Physics Simulation
#             sqnr = simulate_physical_recon(group, s_list, l_list, mrr_cap=cap, noise_sigma=noise_sigma)
            
#             k_list.append(k)
#             sqnr_list.append(sqnr)
            
#         avg_k = np.mean(k_list)
#         avg_sqnr = np.mean(sqnr_list)
#         saving = (8 - avg_k) / 8.0 * 100
        
#         results.append({
#             "Cap": cap,
#             "Label": label,
#             "Avg_K": avg_k,
#             "Avg_SQNR": avg_sqnr,
#             "Saving": saving
#         })

#     df = pd.DataFrame(results)
    
#     # 3. 打印结果表
#     print("\n" + "="*60)
#     print(f"Simulation Results (Model: ResNet18, Noise: {noise_sigma*100}%)")
#     print("="*60)
#     print(df[['Label', 'Avg_K', 'Saving', 'Avg_SQNR']].round(2).to_string(index=False))
#     print("="*60)
    
#     # 4. 绘图
#     sns.set_theme(style="whitegrid")
#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     # 柱状图：压缩收益 (Avg K)
#     # 使用双轴：左轴 K 值，右轴 SQNR
#     color = 'tab:blue'
#     ax1.set_xlabel('MRR Capability (L_max)', fontsize=12)
#     ax1.set_ylabel('Average Slices (K) [Lower is Better]', color=color, fontsize=12)
#     bp = sns.barplot(data=df, x='Label', y='Avg_K', ax=ax1, color='skyblue', alpha=0.6, edgecolor='black')
#     ax1.tick_params(axis='y', labelcolor=color)
#     ax1.set_ylim(0, 9)

#     # 在柱子上标注 Saving
#     for i, p in enumerate(bp.patches):
#         height = p.get_height()
#         ax1.text(p.get_x() + p.get_width() / 2., height + 0.1, 
#                  f"-{df.iloc[i]['Saving']:.1f}%", 
#                  ha="center", color='black', fontsize=10, fontweight='bold')

#     # 折线图：物理精准度 (SQNR)
#     ax2 = ax1.twinx()  
#     color = 'tab:red'
#     ax2.set_ylabel('Physical SQNR (dB) [Higher is Better]', color=color, fontsize=12)
#     sns.lineplot(data=df, x='Label', y='Avg_SQNR', ax=ax2, color=color, marker='o', linewidth=2.5, markersize=8)
#     ax2.tick_params(axis='y', labelcolor=color)
#     ax2.set_ylim(20, 60)
    
#     # 标注 Ours
#     # 找到 L=5 的索引
#     idx_ours = 3
#     ax2.annotate('Sweet Spot\n(High Robustness)', xy=(idx_ours, df.iloc[idx_ours]['Avg_SQNR']), 
#                  xytext=(idx_ours, df.iloc[idx_ours]['Avg_SQNR'] + 10),
#                  arrowprops=dict(facecolor='black', shrink=0.05),
#                  ha='center', fontsize=11, fontweight='bold', color='darkred')

#     plt.title('Comparison of Compression vs. Robustness on ResNet-18 Weights', fontsize=14)
#     plt.tight_layout()
#     plt.savefig('output/real_model_lmax_analysis.png')
#     print("\nPlot saved to 'output/real_model_lmax_analysis.png'")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# 防止 OpenMP 冲突（针对本地 PyTorch 环境）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 1. 核心算法 A: 自适应位致密编码 (Ours)
# ==========================================
def encode_msb_priority(group, max_k=8, mrr_cap=4):
    """
    自适应编码：优先剥离 MSB，利用 MRR 的非标准能级
    """
    original = group.astype(np.float64)
    target = np.abs(original).astype(np.int64)
    
    if np.sum(target) == 0:
        return 0, [], []

    residues = target.copy()
    slices_s = []
    slices_l = []
    
    # 硬件位宽 (用于确定搜索窗口)
    hardware_bit_width = int(np.floor(np.log2(mrr_cap))) + 1

    for k in range(1, max_k + 1):
        max_val = np.max(residues)
        if max_val == 0: 
            break
            
        j_msb = int(np.floor(np.log2(max_val)))
        
        # 搜索窗口
        search_start = max(0, j_msb - hardware_bit_width + 1)
        search_end = j_msb 
        
        best_s = -1
        min_cost = float('inf')
        best_levels = None
        
        for s_candidate in range(search_start, search_end + 1):
            levels = residues >> s_candidate
            levels = np.clip(levels, 0, mrr_cap)
            
            current_comp = levels << s_candidate
            temp_residue = residues - current_comp
            
            # 代价函数：残差能量 + 负惩罚（奖励大 S）
            energy_cost = np.sum(temp_residue) 
            penalty = -s_candidate * 0.1 
            total_cost = energy_cost + penalty
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_s = s_candidate
                best_levels = levels

        if best_levels is None:
             best_s = 0
             best_levels = np.clip(residues, 0, mrr_cap)

        slices_s.append(best_s)
        slices_l.append(best_levels)
        
        current_components = best_levels << best_s
        residues -= current_components
        
    return len(slices_s), slices_s, slices_l

# ==========================================
# 2. 核心算法 B: 硬切片 (Hard Slicing)
# ==========================================
def encode_fixed_slicing(group, bits_per_slice=2):
    """
    硬切分：不跳过零，固定切分
    2-bit -> K=8
    4-bit -> K=4
    """
    target = np.abs(group).astype(np.int64)
    slices_s = []
    slices_l = []
    
    total_bits = 16
    num_slices = total_bits // bits_per_slice
    mask = (1 << bits_per_slice) - 1
    
    for i in range(num_slices):
        s = i * bits_per_slice
        levels = (target >> s) & mask
        
        slices_s.append(s)
        slices_l.append(levels)
        
    return num_slices, slices_s, slices_l

# ==========================================
# 3. 物理层仿真 (含硬编码支持)
# ==========================================
def simulate_physical_recon(original, slices_s, slices_l, mrr_cap, noise_sigma=0.01):
    target = np.abs(original)
    recon_val = np.zeros_like(target, dtype=np.float64)
    
    for s, l_ideal in zip(slices_s, slices_l):
        # 归一化：将能级映射到 [0, 1] 透射率区间
        t_ideal = l_ideal.astype(np.float64) / mrr_cap
        
        # 注入噪声
        noise = np.random.normal(0, noise_sigma, size=t_ideal.shape)
        t_noisy = t_ideal + noise
        
        # 物理还原
        l_noisy = t_noisy * mrr_cap
        recon_val += l_noisy * (2.0 ** s)
        
    err = target - recon_val
    sig_pow = np.sum(target ** 2)
    err_pow = np.sum(err ** 2)
    
    if err_pow == 0: return 99.9
    return 10 * np.log10(sig_pow / err_pow)

# ==========================================
# 4. 数据生成 (模拟 ResNet 权重分布)
# ==========================================
def generate_mock_weights(n_samples=2000, wdm_size=9):
    print("生成模拟权重数据 (Gaussian Distribution)...")
    raw = np.random.randn(n_samples * wdm_size)
    scale = 32767.0 / np.max(np.abs(raw))
    quantized = np.round(raw * scale).astype(np.int32)
    return quantized.reshape(-1, wdm_size)

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 准备数据
    weights = generate_mock_weights(n_samples=2000, wdm_size=9)
    
    # 2. 定义对比配置
    configs = [
        {'type': 'Fixed', 'bits': 2, 'cap': 3,  'label': 'Hard 2-bit\n(L=4)'}, 
        {'type': 'Adaptive', 'cap': 3, 'label': 'Adaptive\n(L=4)'},
        {'type': 'Adaptive', 'cap': 4, 'label': 'Ours\n(L=5)'},     
        {'type': 'Adaptive', 'cap': 5, 'label': 'Adaptive\n(L=6)'},
        {'type': 'Adaptive', 'cap': 6, 'label': 'Adaptive\n(L=7)'},
        {'type': 'Fixed',    'bits': 4, 'cap': 15, 'label': 'Hard 4-bit\n(L=16)'},
    ]
    
    # 3. 定义噪声环境
    noise_levels = [0.005, 0.01, 0.02]
    
    results_k = []
    results_snr = []
    
    print(f"开始仿真 ({len(configs)} 种配置 x {len(noise_levels)} 种噪声环境)...")
    
    for cfg in tqdm(configs):
        # 获取 K 值
        k_list = []
        slices_cache = []
        
        for group in weights:
            if cfg['type'] == 'Adaptive':
                k, ss, sl = encode_msb_priority(group, max_k=8, mrr_cap=cfg['cap'])
            else:
                k, ss, sl = encode_fixed_slicing(group, bits_per_slice=cfg['bits'])
            k_list.append(k)
            slices_cache.append((ss, sl))
            
        avg_k = np.mean(k_list)
        results_k.append({
            'Label': cfg['label'],
            'Avg_K': avg_k,
            'Type': cfg['type']
        })
        
        # 在不同噪声下计算 SNR
        for noise in noise_levels:
            snr_list = []
            for i, group in enumerate(weights):
                ss, sl = slices_cache[i]
                snr = simulate_physical_recon(group, ss, sl, mrr_cap=cfg['cap'], noise_sigma=noise)
                snr_list.append(snr)
            
            results_snr.append({
                'Label': cfg['label'],
                'Noise': f"{noise*100:.1f}%",
                'Avg_SQNR': np.mean(snr_list)
            })

    # ==========================================
    # 6. 分两张图显示
    # ==========================================
    sns.set_theme(style="whitegrid")
    
    # --- 图 1: K 消耗 (Bar Chart) ---
    plt.figure(figsize=(10, 6))
    df_k = pd.DataFrame(results_k)
    
    # 创建颜色映射
    colors = []
    for label in df_k['Label']:
        if 'Ours' in label:
            colors.append('dodgerblue')
        elif 'Hard' in label:
            colors.append('gray')
        else:
            colors.append('skyblue')
    
    bars = plt.bar(df_k['Label'], df_k['Avg_K'], color=colors, edgecolor='black', alpha=0.8)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title('Metric 1: Processing Latency (Avg. K)', fontsize=14, fontweight='bold')
    plt.ylabel('Time Steps (K) [Lower is Better]', fontsize=12)
    plt.ylim(0, 9)
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.8, label='Hard Slicing (Baseline)'),
        Patch(facecolor='skyblue', alpha=0.8, label='Adaptive (Other)'),
        # Patch(facecolor='dodgerblue', alpha=0.8, label='Ours (Best Trade-off)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('k_consumption_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- 图 2: SNR 鲁棒性 (Multi-line Chart) ---
    plt.figure(figsize=(10, 6))
    df_snr = pd.DataFrame(results_snr)
    
    # 获取唯一的标签顺序（与图1保持一致）
    label_order = df_k['Label'].tolist()
    
    # 为每个噪声级别绘制线图
    for noise_level in df_snr['Noise'].unique():
        noise_data = df_snr[df_snr['Noise'] == noise_level]
        # 按图1的顺序排序
        noise_data = noise_data.set_index('Label').reindex(label_order).reset_index()
        
        plt.plot(noise_data['Label'], noise_data['Avg_SQNR'], 
                marker='o', linewidth=2.5, markersize=8, label=noise_level)
    
    plt.title('Metric 2: Physical Robustness (SQNR)', fontsize=14, fontweight='bold')
    plt.ylabel('Reconstruction SQNR (dB) [Higher is Better]', fontsize=12)
    plt.xlabel('Configuration', fontsize=12)
    plt.ylim(20, 60)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Noise Level', loc='lower right')
    
    # 高亮标注 Ours 位置
    ours_label = 'Ours\n(L=5)'
    if ours_label in label_order:
        ours_index = label_order.index(ours_label)
        # 获取所有噪声级别下 Ours 的 SNR 值
        ours_snr_values = []
        for noise_level in df_snr['Noise'].unique():
            value = df_snr[(df_snr['Label'] == ours_label) & (df_snr['Noise'] == noise_level)]['Avg_SQNR']
            if not value.empty:
                ours_snr_values.append(value.values[0])
        
        if ours_snr_values:
            # 取中值位置
            avg_snr = np.mean(ours_snr_values)
            
            # 在 Ours 位置添加标注
            # plt.annotate('Best Trade-off', 
            #             xy=(ours_index, avg_snr), 
            #             xytext=(ours_index, avg_snr + 8),
            #             arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
            #             ha='center', fontsize=11, color='darkred', fontweight='bold',
            #             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('snr_robustness_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印简要数据
    print("\n" + "="*50)
    print("处理延迟数据 (K 值):")
    print("="*50)
    print(df_k[['Label', 'Avg_K']].to_string(index=False))
    
    print("\n" + "="*50)
    print("信噪比数据 (1.0% 噪声水平):")
    print("="*50)
    df_snr_1pct = df_snr[df_snr['Noise'] == '1.0%'][['Label', 'Avg_SQNR']]
    print(df_snr_1pct.to_string(index=False))
    
    print("\n绘图完成！已保存为:")
    print("1. k_consumption_plot.png - 处理延迟图表")
    print("2. snr_robustness_plot.png - 信噪比鲁棒性图表")