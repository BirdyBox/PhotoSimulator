# import torch
# import torchvision.models as models
# import numpy as np
# import pandas as pd
# import itertools
# from tqdm import tqdm
# import torch.nn as nn

# # ==========================================
# # 1. 全局配置与算法参数
# # ==========================================

# # --- 实验设置 ---
# WDM_GROUP_SIZE = 8          # M: 波分复用组大小
# SAMPLE_COUNT_PER_LAYER = 2048 # 每层随机采样多少组进行测试 (设为 -1 则跑全量，极慢)

# # --- 算法阈值 (ACC 指标) ---
# TARGET_SQNR_DB = 35.0       # 目标信噪比 (dB)，超过即停止增加 K
# MAX_ABS_ERR_TOL = 32        # 绝对误差容忍度 (忽略低位噪声)

# # --- 硬件约束 ---
# MAX_K_LIMIT = 4             # 物理行上限 (Baseline)
# MRR_LEVEL_CAP = 4           # MRR 最大能级 (100)
# MIN_S_INTERVAL = 2          # S 之间的最小间距 (正交排斥域)
# SEARCH_SPACE = list(range(13)) # S 候选范围 [0...12]

# # ==========================================
# # 2. 算法核心：预计算与正交搜索
# # ==========================================

# # 预计算所有合法的 S 组合，避免循环中重复计算
# VALID_COMBOS_CACHE = {}

# def precompute_combos():
#     """预先生成所有满足间距约束的 S 组合"""
#     print(f"正在预计算合法 S 组合 (间距 >= {MIN_S_INTERVAL})...")
#     for k in range(1, MAX_K_LIMIT + 1):
#         combos = []
#         for c in itertools.combinations(SEARCH_SPACE, k):
#             # 检查间距
#             is_valid = True
#             for i in range(len(c) - 1):
#                 if c[i+1] - c[i] < MIN_S_INTERVAL:
#                     is_valid = False
#                     break
#             if is_valid:
#                 combos.append(c)
#         VALID_COMBOS_CACHE[k] = combos
#         print(f"  K={k}: 找到 {len(combos)} 种合法组合")

# def evaluate_metrics(original, reconstructed):
#     """计算 SQNR 和 MaxErr"""
#     error = original - reconstructed
#     max_err = np.max(np.abs(error))
    
#     sig_pow = np.sum(original ** 2)
#     err_pow = np.sum(error ** 2)
    
#     if err_pow == 0:
#         sqnr = 99.9 # Inf
#     elif sig_pow == 0:
#         sqnr = 0.0
#     else:
#         sqnr = 10 * np.log10(sig_pow / err_pow)
#     return sqnr, max_err

# def fit_with_s_combo(residues_in, s_combo):
#     """
#     使用固定的 S 组合进行 Floor 拟合
#     """
#     # 必须复制，避免修改原始数据
#     residues = residues_in.copy()
#     recon = np.zeros_like(residues)
    
#     # 从大 S 到小 S 排序
#     sorted_s = sorted(s_combo, reverse=True)
    
#     for s in sorted_s:
#         # Floor 拟合 + Clip
#         levels = np.floor(residues / (2.0 ** s))
#         levels = np.clip(levels, 0, MRR_LEVEL_CAP)
        
#         comp = levels * (2.0 ** s)
#         recon += comp
#         residues -= comp
        
#     return recon

# def optimal_orthogonal_encode(group):
#     """
#     对单个组进行自适应 K 压缩
#     """
#     target = np.abs(group).astype(float)
#     if np.sum(target) == 0:
#         return 0, 99.9, 0 # K=0 for all zeros (Gate OFF)

#     best_recon = np.zeros_like(target)
    
#     # 动态尝试 K = 1 到 MAX
#     for k in range(1, MAX_K_LIMIT + 1):
#         # 遍历该 K 下所有预计算的组合
#         k_best_sqnr = -1
#         k_best_recon = None
        
#         # 这里的循环是性能瓶颈，Sampling 很有必要
#         combos = VALID_COMBOS_CACHE[k]
        
#         for combo in combos:
#             recon = fit_with_s_combo(target, combo)
#             # 快速计算 SQNR (内联以提速)
#             err = target - recon
#             err_pow = np.sum(err**2)
#             if err_pow == 0:
#                 cur_sqnr = 99.9
#             else:
#                 sig_pow = np.sum(target**2)
#                 cur_sqnr = 10 * np.log10(sig_pow / err_pow)
            
#             if cur_sqnr > k_best_sqnr:
#                 k_best_sqnr = cur_sqnr
#                 k_best_recon = recon
        
#         # 检查是否达标
#         _, max_err = evaluate_metrics(target, k_best_recon)
        
#         if k_best_sqnr >= TARGET_SQNR_DB or max_err <= MAX_ABS_ERR_TOL:
#             return k, k_best_sqnr, max_err
            
#     # 达到上限仍未达标
#     return MAX_K_LIMIT, k_best_sqnr, max_err

# # ==========================================
# # 3. 实验主流程
# # ==========================================

# def run_resnet18_experiment():
#     # 1. 初始化
#     precompute_combos()
#     print("\n正在加载 ResNet18 模型...")
#     model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
#     # 提取卷积层
#     layers = []
#     for name, m in model.named_modules():
#         if isinstance(m, torch.nn.Conv2d):
#             layers.append((name, m))
            
#     print(f"共找到 {len(layers)} 个卷积层。开始测试...")
#     print(f"采样策略: 每层随机抽取 {SAMPLE_COUNT_PER_LAYER} 个组进行测试\n")
    
#     results = []
    
#     # 2. 逐层处理
#     for layer_name, layer in tqdm(layers):
#         # 量化权重
#         w_float = layer.weight.detach().numpy()
#         w_max = np.max(np.abs(w_float))
#         scale = 32767.0 / w_max if w_max > 0 else 1.0
#         w_int16 = np.round(w_float * scale).astype(np.int16)
        
#         # 展平并分组
#         flat = w_int16.flatten()
#         pad_len = (WDM_GROUP_SIZE - (len(flat) % WDM_GROUP_SIZE)) % WDM_GROUP_SIZE
#         padded = np.pad(flat, (0, pad_len), 'constant')
#         all_groups = padded.reshape(-1, WDM_GROUP_SIZE)
        
#         # 采样 (如果数据量太大)
#         total_groups = len(all_groups)
#         if SAMPLE_COUNT_PER_LAYER > 0 and total_groups > SAMPLE_COUNT_PER_LAYER:
#             indices = np.random.choice(total_groups, SAMPLE_COUNT_PER_LAYER, replace=False)
#             test_groups = all_groups[indices]
#         else:
#             test_groups = all_groups
            
#         # 运行压缩算法
#         k_stats = {0:0, 1:0, 2:0, 3:0, 4:0}
#         sqnr_list = []
        
#         for group in test_groups:
#             k_used, sqnr, _ = optimal_orthogonal_encode(group)
#             k_stats[k_used] += 1
#             sqnr_list.append(sqnr)
            
#         # 统计本层数据
#         avg_k = sum([k * count for k, count in k_stats.items()]) / len(test_groups)
#         avg_sqnr = np.mean(sqnr_list)
        
#         # 计算节省比例 (相对于 Baseline K=4)
#         # 如果 K=0 (全0组)，节省也是 4
#         baseline_k = MAX_K_LIMIT
#         saving_ratio = (baseline_k - avg_k) / baseline_k
        
#         results.append({
#             "Layer": layer_name,
#             "Total_Params": len(flat),
#             "Samples": len(test_groups),
#             "Avg_K": round(avg_k, 2),
#             "Avg_SQNR": round(avg_sqnr, 2),
#             "MRR_Saving": f"{saving_ratio*100:.1f}%",
#             "K0_Ratio": f"{k_stats[0]/len(test_groups)*100:.1f}%", # 稀疏度
#             "K1_Ratio": f"{k_stats[1]/len(test_groups)*100:.1f}%",
#             "K2_Ratio": f"{k_stats[2]/len(test_groups)*100:.1f}%",
#             "K3_Ratio": f"{k_stats[3]/len(test_groups)*100:.1f}%",
#             "K4_Ratio": f"{k_stats[4]/len(test_groups)*100:.1f}%"
#         })
        
#     # 3. 生成报告
#     df = pd.DataFrame(results)
#     print("\n" + "="*80)
#     print("ResNet18 Bit-Density 压缩测试报告")
#     print("="*80)
#     print(df[["Layer", "Avg_K", "Avg_SQNR", "MRR_Saving", "K1_Ratio", "K4_Ratio"]].to_string())
    
#     # 保存
#     df.to_csv("resnet18_bit_density_report.csv", index=False)
#     print("\n详细报告已保存至 resnet18_bit_density_report.csv")
    
#     # 总体统计
#     total_saving = df["MRR_Saving"].str.rstrip('%').astype(float).mean()
#     print(f"\n[结论] 整个模型的平均 MRR 节省率: {total_saving:.2f}%")
#     print(f"[结论] 平均 SQNR 质量: {df['Avg_SQNR'].mean():.2f} dB")
# class UniversalCompressor:
#     def __init__(self, wdm_group_size=8, algorithm_func=None):
#         self.wdm_group_size = wdm_group_size
#         self.algorithm_func = algorithm_func  # 传入之前的 optimal_orthogonal_encode 函数
#         self.results = []

#     def _process_weight(self, weight_tensor, layer_name):
#         """
#         通用权重处理核心：量化 -> 分组 -> 压缩 -> 统计
#         """
#         # 1. 提取与量化 (Float -> Int16)
#         w_float = weight_tensor.detach().cpu().numpy()
#         w_max = np.max(np.abs(w_float))
        
#         # 避免全0层导致除以0
#         if w_max == 0:
#             scale = 1.0
#         else:
#             scale = 32767.0 / w_max
            
#         w_int16 = np.round(w_float * scale).astype(np.int16)

#         # 2. 展平与填充 (Flatten & Pad)
#         # 无论是 4D Conv 还是 2D Linear，一旦 Flatten，物理上都是比特流
#         flat = w_int16.flatten()
        
#         # 填充以适配 WDM 组大小
#         pad_len = (self.wdm_group_size - (len(flat) % self.wdm_group_size)) % self.wdm_group_size
#         padded = np.pad(flat, (0, pad_len), 'constant')
        
#         # 3. 分组
#         groups = padded.reshape(-1, self.wdm_group_size)
        
#         # 4. 采样 (为了速度，比如只测 2000 组)
#         # 在这里可以复用您之前的采样逻辑
#         sample_size = 2048
#         if len(groups) > sample_size:
#             indices = np.random.choice(len(groups), sample_size, replace=False)
#             test_groups = groups[indices]
#         else:
#             test_groups = groups

#         # 5. 运行核心算法
#         k_counts = []
#         sqnrs = []
        
#         for group in test_groups:
#             # 调用您现有的核心函数
#             k, sqnr, _ = self.algorithm_func(group)
#             k_counts.append(k)
#             sqnrs.append(sqnr)

#         return {
#             "Layer": layer_name,
#             "Type": "Linear" if len(weight_tensor.shape)==2 else "Conv2d",
#             "Shape": str(tuple(weight_tensor.shape)),
#             "Avg_K": np.mean(k_counts),
#             "Avg_SQNR": np.mean(sqnrs),
#             "Saving": (4.0 - np.mean(k_counts)) / 4.0  # 假设 Baseline K=4
#         }

#     def compress_model(self, model):
#         """
#         遍历模型所有层，自动识别支持的类型
#         """
#         self.results = []
#         print(f"正在分析模型结构...")
        
#         # 遍历所有模块
#         for name, module in tqdm(model.named_modules()):
#             # 策略：只处理有 weight 属性的层
#             # 增加对 Linear 的支持
#             if isinstance(module, (nn.Conv2d, nn.Linear)):
#                 # 跳过没有权重的层 (极少见)
#                 if not hasattr(module, 'weight') or module.weight is None:
#                     continue
                
#                 # 某些 Linear 层可能用于最后分类，参数量巨大，需不需要跳过看需求
#                 # 这里默认全部处理
#                 stats = self._process_weight(module.weight, name)
#                 self.results.append(stats)
                
#         return self.results

#     def print_report(self):
#         import pandas as pd
#         df = pd.DataFrame(self.results)
#         print("\n=== 通用模型压缩报告 ===")
#         # 格式化输出
#         if not df.empty:
#             df['Saving'] = df['Saving'].apply(lambda x: f"{x*100:.1f}%")
#             df['Avg_SQNR'] = df['Avg_SQNR'].apply(lambda x: f"{x:.2f}")
#             df['Avg_K'] = df['Avg_K'].apply(lambda x: f"{x:.2f}")
#             print(df[['Layer', 'Type', 'Shape', 'Avg_K', 'Avg_SQNR', 'Saving']].to_string())
#         else:
#             print("未找到支持的层 (Conv2d/Linear)")
            
# if __name__ == "__main__":
#     run_resnet18_experiment()
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
import itertools
import os
import time
from tqdm import tqdm

# ==========================================
# 1. 核心算法逻辑 (复用之前的正交搜索算法)
# ==========================================

# --- 全局配置 ---
TARGET_SQNR_DB = 35.0
MAX_ABS_ERR_TOL = 32
MAX_K_LIMIT = 4
MRR_LEVEL_CAP = 4
MIN_S_INTERVAL = 2
SEARCH_SPACE = list(range(13))
VALID_COMBOS_CACHE = {}

def precompute_combos():
    """预计算合法 S 组合"""
    if VALID_COMBOS_CACHE: return # 避免重复计算
    # print("正在初始化算法查找表...")
    for k in range(1, MAX_K_LIMIT + 1):
        combos = []
        for c in itertools.combinations(SEARCH_SPACE, k):
            is_valid = True
            for i in range(len(c) - 1):
                if c[i+1] - c[i] < MIN_S_INTERVAL:
                    is_valid = False
                    break
            if is_valid:
                combos.append(c)
        VALID_COMBOS_CACHE[k] = combos

def fit_with_s_combo(residues_in, s_combo):
    residues = residues_in.copy()
    recon = np.zeros_like(residues)
    sorted_s = sorted(s_combo, reverse=True)
    for s in sorted_s:
        levels = np.floor(residues / (2.0 ** s))
        levels = np.clip(levels, 0, MRR_LEVEL_CAP)
        comp = levels * (2.0 ** s)
        recon += comp
        residues -= comp
    return recon

def optimal_orthogonal_encode(group):
    """单组压缩核心函数"""
    target = np.abs(group).astype(float)
    if np.sum(target) == 0:
        return 0, 99.9, 0 

    best_recon = np.zeros_like(target)
    
    for k in range(1, MAX_K_LIMIT + 1):
        k_best_sqnr = -1
        k_best_recon = None
        
        combos = VALID_COMBOS_CACHE[k]
        for combo in combos:
            recon = fit_with_s_combo(target, combo)
            err = target - recon
            err_pow = np.sum(err**2)
            if err_pow == 0:
                cur_sqnr = 99.9
            else:
                sig_pow = np.sum(target**2)
                cur_sqnr = 10 * np.log10(sig_pow / err_pow)
            
            if cur_sqnr > k_best_sqnr:
                k_best_sqnr = cur_sqnr
                k_best_recon = recon
        
        # 简单检查 MaxErr
        max_err = np.max(np.abs(target - k_best_recon))
        
        if k_best_sqnr >= TARGET_SQNR_DB or max_err <= MAX_ABS_ERR_TOL:
            return k, k_best_sqnr, max_err
            
    return MAX_K_LIMIT, k_best_sqnr, max_err

# ==========================================
# 2. 通用压缩器 (UniversalCompressor)
# ==========================================

class UniversalCompressor:
    def __init__(self, wdm_group_size=8, algorithm_func=None):
        self.wdm_group_size = wdm_group_size
        self.algorithm_func = algorithm_func
        self.results = []

    def _process_weight(self, weight_tensor, layer_name):
        # 1. 量化
        w_float = weight_tensor.detach().cpu().numpy()
        w_max = np.max(np.abs(w_float))
        if w_max == 0: scale = 1.0
        else: scale = 32767.0 / w_max
        w_int16 = np.round(w_float * scale).astype(np.int16)

        # 2. 展平与填充
        flat = w_int16.flatten()
        pad_len = (self.wdm_group_size - (len(flat) % self.wdm_group_size)) % self.wdm_group_size
        padded = np.pad(flat, (0, pad_len), 'constant')
        
        # 3. 分组
        groups = padded.reshape(-1, self.wdm_group_size)
        
        # 4. 采样 (每层最多测 1024 组以节省时间，可根据需求调整)
        sample_size = 1024*16
        if len(groups) > sample_size:
            indices = np.random.choice(len(groups), sample_size, replace=False)
            test_groups = groups[indices]
        else:
            test_groups = groups

        # 5. 运行算法
        k_counts = []
        sqnrs = []
        
        for group in test_groups:
            k, sqnr, _ = self.algorithm_func(group)
            k_counts.append(k)
            sqnrs.append(sqnr)

        avg_k = np.mean(k_counts)
        return {
            "Layer": layer_name,
            "Type": "Linear" if len(weight_tensor.shape)==2 else "Conv2d",
            "Params": len(flat),
            "Avg_K": avg_k,
            "Avg_SQNR": np.mean(sqnrs),
            "Saving_Ratio": (4.0 - avg_k) / 4.0 
        }

    def compress_model(self, model, model_name="Unknown"):
        self.results = []
        # print(f"正在分析模型: {model_name} ...")
        
        # 遍历所有模块
        for name, module in tqdm(model.named_modules(), desc=f"Scanning {model_name}", leave=False):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if not hasattr(module, 'weight') or module.weight is None:
                    continue
                
                stats = self._process_weight(module.weight, name)
                stats['Model'] = model_name # 增加一列模型名称
                self.results.append(stats)
                
        return self.results

# ==========================================
# 3. 自动化测试台 (Benchmark Runner)
# ==========================================

def get_model_zoo():
    """构建需要测试的模型字典"""
    zoo = {}
    
    print("正在准备模型库...")
    # 1. 经典 CNN
    try: zoo['AlexNet'] = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    except: pass
    
    try: zoo['VGG16'] = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    except: pass
    
    try: zoo['ResNet18'] = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except: pass
    
    try: zoo['ResNet50'] = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except: pass
    
    try: zoo['MobileNetV2'] = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    except: pass
    
    try: zoo['GoogleNet'] = models.googlenet(weights=models.Googlenet_Weights.DEFAULT)
    except: pass

    # 2. Transformer (Bert)
    # 需要 pip install transformers
    try:
        from transformers import BertModel
        # 使用 tiny bert 避免下载太慢，或者 standard 'bert-base-uncased'
        print("尝试加载 BERT (需要联网)...")
        zoo['BERT-Base'] = BertModel.from_pretrained('bert-base-uncased')
    except ImportError:
        print("[Warn] 未安装 transformers 库，跳过 BERT 测试。")
    except Exception as e:
        print(f"[Warn] BERT 加载失败: {e}")

    # 3. YOLO (通常需要 ultralytics 库，这里用标准库无法直接加载)
    # 建议手动加载 weights 或者忽略
    
    return zoo

def run_unattended_benchmark():
    # 1. 准备环境
    precompute_combos() # 初始化算法缓存
    compressor = UniversalCompressor(wdm_group_size=8, algorithm_func=optimal_orthogonal_encode)
    
    models_to_test = get_model_zoo()
    print(f"\n成功加载 {len(models_to_test)} 个模型待测试: {list(models_to_test.keys())}")
    print("="*60)
    print("开始全自动测试。结果将实时保存到 CSV。请勿关闭窗口。")
    print("="*60)

    # 2. 全局结果容器
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detail_report_filename = f"benchmark_layers_{timestamp}.csv"
    summary_report_filename = f"benchmark_summary_{timestamp}.csv"
    
    all_layer_stats = []
    summary_stats = []

    # 3. 循环测试
    for model_name, model in models_to_test.items():
        start_time = time.time()
        try:
            # 运行压缩
            model.eval() # 设为评估模式
            layer_stats = compressor.compress_model(model, model_name)
            
            # 收集数据
            all_layer_stats.extend(layer_stats)
            
            # 计算该模型的整体摘要
            df_curr = pd.DataFrame(layer_stats)
            if not df_curr.empty:
                # 加权平均 (按参数量加权，这样更能反映整体硬件节省)
                total_params = df_curr['Params'].sum()
                weighted_saving = (df_curr['Saving_Ratio'] * df_curr['Params']).sum() / total_params
                avg_sqnr = df_curr['Avg_SQNR'].mean()
                
                summary = {
                    "Model": model_name,
                    "Total_Params (M)": round(total_params / 1e6, 2),
                    "Layers_Processed": len(df_curr),
                    "Weighted_Saving": f"{weighted_saving*100:.2f}%",
                    "Avg_SQNR": f"{avg_sqnr:.2f}",
                    "Time_Sec": round(time.time() - start_time, 1)
                }
                summary_stats.append(summary)
                print(f"完成 {model_name}: 节省 {summary['Weighted_Saving']} | SQNR {summary['Avg_SQNR']} | 耗时 {summary['Time_Sec']}s")
            
            # --- 实时存档 (防止断电) ---
            pd.DataFrame(all_layer_stats).to_csv(detail_report_filename, index=False)
            pd.DataFrame(summary_stats).to_csv(summary_report_filename, index=False)
            
        except Exception as e:
            print(f"[Error] 模型 {model_name} 测试出错: {e}")
            continue

    print("\n" + "="*60)
    print("所有测试已完成！")
    print(f"1. 层级详细报告: {detail_report_filename}")
    print(f"2. 模型摘要报告: {summary_report_filename}")
    print("="*60)
    
    # 打印最终摘要表
    if summary_stats:
        print(pd.DataFrame(summary_stats).to_string())

if __name__ == "__main__":
    run_unattended_benchmark()