import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import itertools
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 1. 模型工厂 (支持多种模型)
# ==========================================
def get_model(model_name, pretrained=True):
    print(f"正在加载模型: {model_name} (Pretrained={pretrained})...")
    
    # 字典映射：名称 -> 加载函数
    # 您可以在这里添加更多 torchvision 支持的模型
    model_dict = {
        'alexnet': models.alexnet,
        'vgg16': models.vgg16,
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'mobilenet_v2': models.mobilenet_v2,
        'googlenet': models.googlenet,
        'densenet121': models.densenet121,
    }
    
    if model_name not in model_dict:
        raise ValueError(f"不支持的模型名称: {model_name}. 请从 {list(model_dict.keys())} 中选择。")
    
    # 获取权重枚举类 (新版 Torchvision 写法)
    weights_enum = None
    if pretrained:
        # 动态获取权重对象，例如 models.AlexNet_Weights.IMAGENET1K_V1
        # 这里为了通用简单，使用 'DEFAULT' (指向最新最优权重)
        try:
            # 这种写法兼容性最好，直接传 weights参数
            if model_name == 'alexnet': weights = models.AlexNet_Weights.DEFAULT
            elif model_name == 'vgg16': weights = models.VGG16_Weights.DEFAULT
            elif model_name == 'resnet18': weights = models.ResNet18_Weights.DEFAULT
            elif model_name == 'resnet50': weights = models.ResNet50_Weights.DEFAULT
            elif model_name == 'mobilenet_v2': weights = models.MobileNet_V2_Weights.DEFAULT
            elif model_name == 'googlenet': weights = models.Googlenet_Weights.DEFAULT
            elif model_name == 'densenet121': weights = models.DenseNet121_Weights.DEFAULT
            else: weights = 'DEFAULT' # 尝试自动推断
        except:
            weights = 'DEFAULT' # Fallback
            
        return model_dict[model_name](weights=weights)
    else:
        return model_dict[model_name](weights=None)

# ==========================================
# 2. 核心压缩算法 (Bit-Density Algorithm 2)
# ==========================================
# 配置
MAX_K_LIMIT = 4
MRR_LEVEL_CAP = 4
MIN_S_INTERVAL = 2
SEARCH_SPACE = list(range(13))
TARGET_SQNR_DB = 35.0
MAX_ABS_ERR_TOL = 32

# 缓存
VALID_COMBOS_CACHE = {}

def precompute_combos():
    if VALID_COMBOS_CACHE: return
    for k in range(1, MAX_K_LIMIT + 1):
        combos = []
        for c in itertools.combinations(SEARCH_SPACE, k):
            is_valid = True
            for i in range(len(c) - 1):
                if c[i+1] - c[i] < MIN_S_INTERVAL:
                    is_valid = False; break
            if is_valid: combos.append(c)
        VALID_COMBOS_CACHE[k] = combos

def fit_with_s_combo(residues_in, s_combo):
    residues = residues_in.copy()
    recon = np.zeros_like(residues)
    for s in sorted(s_combo, reverse=True):
        levels = np.floor(residues / (2.0 ** s))
        levels = np.clip(levels, 0, MRR_LEVEL_CAP)
        comp = levels * (2.0 ** s)
        recon += comp
        residues -= comp
    return recon

def compress_single_group(group):
    # 子进程需重新检查缓存
    if not VALID_COMBOS_CACHE: precompute_combos()
    
    target = np.abs(group).astype(float)
    if np.sum(target) == 0: return np.zeros_like(group)

    best_recon = np.zeros_like(target)
    
    for k in range(1, MAX_K_LIMIT + 1):
        k_best_sqnr = -1
        k_best_recon = None
        
        for combo in VALID_COMBOS_CACHE[k]:
            recon = fit_with_s_combo(target, combo)
            err = target - recon
            err_pow = np.sum(err**2)
            cur_sqnr = 99.9 if err_pow == 0 else 10 * np.log10(np.sum(target**2) / err_pow)
            
            if cur_sqnr > k_best_sqnr:
                k_best_sqnr = cur_sqnr
                k_best_recon = recon
        
        max_err = np.max(np.abs(target - k_best_recon))
        if k_best_sqnr >= TARGET_SQNR_DB or max_err <= MAX_ABS_ERR_TOL:
            return k_best_recon * np.sign(group)
            
    return k_best_recon * np.sign(group)

def process_layer_weights(w_flat, wdm_group_size):
    pad_len = (wdm_group_size - (len(w_flat) % wdm_group_size)) % wdm_group_size
    padded = np.pad(w_flat, (0, pad_len), 'constant')
    groups = padded.reshape(-1, wdm_group_size)
    
    max_workers = os.cpu_count()
    # 针对 AlexNet/VGG 的巨大全连接层，适当增加 chunksize
    chunksize = max(10, len(groups) // (max_workers * 8))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compress_single_group, groups, chunksize=chunksize))
        
    recon_flat = np.concatenate(results)[:len(w_flat)]
    return recon_flat

import numpy as np

# ==========================================
# 核心算法：模式驱动位矩阵覆盖 (Bit-Matrix Pattern Covering)
# ==========================================

def compress_single_group_pattern(group, wdm_group_size=8, max_k=4, target_sqnr=35.0):
    """
    基于位模式匹配的自适应步进压缩算法
    """
    # 1. 预处理：处理补码，提取绝对值矩阵
    original = group.astype(np.float64)
    target = np.abs(original)
    signs = np.sign(original)
    
    if np.sum(target) == 0:
        return np.zeros_like(group)

    residues = target.copy()
    reconstructed = np.zeros_like(target)
    
    # 记录已使用的 K
    for k in range(max_k):
        # A. 扫描位矩阵：找到当前所有残差中最高位的 1 的位置 j
        # 例如 32768 -> j=15, 1 -> j=0
        max_val = np.max(residues)
        if max_val < 1.0: # 已经基本压完了
            break
            
        j_max = int(np.floor(np.log2(max_val)))
        
        # B. 模式识别逻辑：
        # 查看最高位权重及其右侧一位的分布 (Pattern Matching)
        # 我们寻找 '11x' (两个高位连续为1) 或 '10x' 模式
        # 这里的识别是以组内“贡献最大”的那个权重为基准
        top_idx = np.argmax(residues)
        val_at_top = int(residues[top_idx])
        
        # 提取模式位 (检查 j_max 和 j_max-1)
        bit_j = (val_at_top >> j_max) & 1
        bit_j_minus_1 = (val_at_top >> (j_max - 1)) & 1 if j_max > 0 else 0
        
        # 动态决定本行的 S
        if bit_j == 1 and bit_j_minus_1 == 1:
            # 模式 11x：为了同时消除两个高位，S 选在 j_max - 1
            # 此时 L=3 (11) 可以覆盖这两个位
            s_k = j_max - 1
        else:
            # 模式 10x 或孤立位：S 选在 j_max - 2
            # 此时 L=4 (100) 可以完美覆盖 j_max 位
            s_k = j_max - 2
            
        # 边界保护：S 不能小于 0
        s_k = max(0, s_k)

        # C. 执行格点量化 (L=0~4)
        # 在选定的 S 下，计算每个权重的最优能级
        levels = np.round(residues / (2.0**s_k))
        levels = np.clip(levels, 0, 4) # 硬件能级限制
        
        # D. 更新重构值与残差
        current_comp = levels * (2.0**s_k)
        reconstructed += current_comp
        residues -= current_comp
        
        # E. 检查退出标准 (SQNR 或 物理行用尽)
        err = target - reconstructed
        err_pow = np.sum(err**2)
        signal_pow = np.sum(target**2)
        
        cur_sqnr = 99.9 if err_pow == 0 else 10 * np.log10(signal_pow / err_pow)
        
        if cur_sqnr >= target_sqnr:
            break

    # 返回带符号的重构值
    return reconstructed * signs

def precise_stripping_encode(group, max_k=4, mrr_cap=4, target_sqnr=35.0):
    """
    [终极版] 精准剥离 + LSB优先 + 自动位模式匹配
    """
    # 1. 预处理：确保使用 int64 进行位运算
    original = group.astype(np.float64)
    target = np.abs(original).astype(np.int64)
    signs = np.sign(original)
    
    if np.sum(target) == 0:
        return np.zeros_like(group)

    residues = target.copy()
    reconstructed = np.zeros_like(target, dtype=np.int64)
    
    # 硬件位宽能力 (例如 4->3 bits)
    # L_max=4 (100) -> log2(4)=2 -> width=3
    hardware_bit_width = int(np.floor(np.log2(mrr_cap))) + 1

    for k in range(1, max_k + 1):
        # A. 锚定最大值 MSB
        max_val = np.max(residues)
        if max_val == 0: break
        
        j_msb = int(np.floor(np.log2(max_val)))
        
        # B. 定义搜索窗口
        # 我们只在 MSB 附近的有限范围内搜索 S
        # 范围：[MSB - 硬件位宽 + 1, MSB]
        # 例如 MSB=4, width=3 -> S range [2, 4]
        search_start = max(0, j_msb - hardware_bit_width + 1)
        search_end = j_msb 
        
        # C. 策略竞争 (Strategy Competition)
        best_s = -1
        min_cost = float('inf')
        best_levels = None
        
        # 关键：遍历所有可能的 S
        for s_candidate in range(search_start, search_end + 1):
            
            # --- 核心差异点：使用位运算进行“精准剥离” ---
            # 1. 计算当前 S 下的能级 (相当于 floor)
            levels = residues >> s_candidate
            
            # 2. 硬件截断 (Clip)
            levels = np.clip(levels, 0, mrr_cap)
            
            # 3. 计算剥离后的新残差
            current_comp = levels << s_candidate
            temp_residue = residues - current_comp
            
            # 4. 代价函数
            # 主要代价：剩余残差的 L1 范数 (或 L2 能量)
            energy_cost = np.sum(temp_residue) 
            
            # 辅助代价：S 的惩罚 (LSB 优先)
            # S 越大，惩罚越大。迫使算法在效果相同时选小 S
            penalty = s_candidate * 0.0001 
            
            total_cost = energy_cost + penalty
            
            # 更新策略
            if total_cost < min_cost:
                min_cost = total_cost
                best_s = s_candidate
                best_levels = levels

        # D. 执行最佳策略
        if best_levels is None: # 兜底保护，虽然理论上不会发生
             best_s = 0
             best_levels = np.clip(residues, 0, mrr_cap)

        current_components = best_levels << best_s
        reconstructed += current_components
        residues -= current_components # 精准减法
        
        # E. 质量检查 (SQNR)
        # 注意：这里我们用 float 计算 SQNR 避免溢出
        err = target.astype(float) - reconstructed.astype(float)
        sig_pow = np.sum(target.astype(float)**2)
        err_pow = np.sum(err**2)
        cur_sqnr = 99.9 if err_pow == 0 else 10 * np.log10(sig_pow / err_pow)
        
        if cur_sqnr >= target_sqnr:
            break

    return reconstructed.astype(np.float64) * signs
# ==========================================
# 适配 ProcessPoolExecutor 的包装器
# ==========================================
# def process_layer_weights_pattern(w_flat, wdm_group_size=8, max_k=4):
#     pad_len = (wdm_group_size - (len(w_flat) % wdm_group_size)) % wdm_group_size
#     padded = np.pad(w_flat, (0, pad_len), 'constant')
#     groups = padded.reshape(-1, wdm_group_size)
    
#     # 使用列表推导式或 Pool
#     # 注意：这里的算法复杂度比暴力搜索低得多，因此 chunksize 可以设大
#     recon_groups = []
#     for g in groups:
#         recon_groups.append(compress_single_group_pattern(g, wdm_group_size, max_k))
        
#     recon_flat = np.concatenate(recon_groups)[:len(w_flat)]
#     return recon_flat
def process_layer_weights_pattern(w_flat, wdm_group_size=8, max_k=4):
    pad_len = (wdm_group_size - (len(w_flat) % wdm_group_size)) % wdm_group_size
    padded = np.pad(w_flat, (0, pad_len), 'constant')
    groups = padded.reshape(-1, wdm_group_size)
    
    # 使用 ProcessPoolExecutor 进行并行加速
    # 注意：precise_stripping_encode 不需要预计算缓存，非常适合并行
    max_workers = os.cpu_count()
    chunksize = max(100, len(groups) // (max_workers * 4)) # 增大块大小减少开销
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 使用 map 自动分发任务
        # 注意：map 需要单独的 worker 函数，lambda 在 pickle 时可能会有问题
        # 这里为了简单直接串行化或者定义一个 helper
        # 由于 Python 多进程的限制，我们直接在这里展开循环，或者使用简单的列表推导式
        # 如果数据量特别大，建议用 tqdm 包装
        
        # 简单并行方案 (如果模型特别大建议用这个)
        results = list(executor.map(precise_stripping_encode, groups, chunksize=chunksize))

    recon_flat = np.concatenate(results)[:len(w_flat)]
    return recon_flat
# ==========================================
# 3. 主逻辑
# ==========================================
def main():
    # --- 配置区域 ---
    MODEL_NAME = "vgg16"  # <=== 修改这里！可选: 'alexnet', 'vgg16', 'mobilenet_v2', 'resnet50'
    OUTPUT_FILE = f"{MODEL_NAME}_compressed_Binary.pth"
    
    print(f"=== 开始处理模型: {MODEL_NAME} ===")
    
    # 1. 加载模型
    model = get_model(MODEL_NAME, pretrained=True)
    precompute_combos()
    
    # 2. 收集需要压缩的层
    # AlexNet/VGG 有大量 Linear 层，ResNet 主要是 Conv2d，这里全部覆盖
    layers_to_process = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 跳过没有权重的层
            if hasattr(module, 'weight') and module.weight is not None:
                layers_to_process.append((name, module))
    
    print(f"共发现 {len(layers_to_process)} 个待压缩层 (Conv2d/Linear)")
    print(f"使用 CPU 核心数: {os.cpu_count()} (并行加速中)")
    
    # 3. 逐层压缩
    for name, module in tqdm(layers_to_process, desc="Compressing"):
        w_device = module.weight.device
        w_float = module.weight.detach().cpu().numpy()
        w_shape = w_float.shape
        
        # 量化
        w_max = np.max(np.abs(w_float))
        scale = 32767.0 / w_max if w_max > 0 else 1.0
        w_int16 = np.round(w_float * scale).astype(np.int16)
        
        # 压缩 (耗时步骤)
        # AlexNet 的 classifier.6 (Linear) 有 4096*1000 参数，这里会跑一会儿
        w_recon_int16_flat = process_layer_weights_pattern(w_int16.flatten(), wdm_group_size=8)
        w_recon_int16 = w_recon_int16_flat.reshape(w_shape)
        
        # 反量化并写回
        w_recon_float = w_recon_int16.astype(np.float32) / scale
        module.weight.data = torch.from_numpy(w_recon_float).to(w_device)
        
    # 4. 保存
    print(f"正在保存至 {OUTPUT_FILE} ...")
    torch.save(model.state_dict(), OUTPUT_FILE)
    print("完成。")

if __name__ == "__main__":
    main()