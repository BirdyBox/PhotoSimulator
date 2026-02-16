import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 1. 模型工厂
# ==========================================
def get_model(model_name, pretrained=True):
    print(f"正在加载模型: {model_name} (Pretrained={pretrained})...")
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
        raise ValueError(f"不支持的模型名称: {model_name}")
    
    weights = 'DEFAULT' if pretrained else None
    return model_dict[model_name](weights=weights)

# ==========================================
# 2. 核心算法：精准剥离 + LSB优先 (Bit-Matrix Precise Stripping)
# ==========================================

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
# 3. 并行处理包装器
# ==========================================

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
# 4. 主逻辑
# ==========================================
def main():
    # --- 配置区域 ---
    MODEL_NAME = "alexnet"  # 可选: 'alexnet', 'vgg16', 'resnet18'
    OUTPUT_FILE = f"{MODEL_NAME}_compressed_Binary.pth"
    
    print(f"=== 开始处理模型: {MODEL_NAME} (精准剥离算法) ===")
    
    # 1. 加载模型
    model = get_model(MODEL_NAME, pretrained=True)
    
    # 2. 收集需要压缩的层
    layers_to_process = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                layers_to_process.append((name, module))
    
    print(f"共发现 {len(layers_to_process)} 个待压缩层")
    print(f"使用 CPU 核心数: {os.cpu_count()}")
    
    # 3. 逐层压缩
    for name, module in tqdm(layers_to_process, desc="Compressing"):
        w_device = module.weight.device
        w_float = module.weight.detach().cpu().numpy()
        w_shape = w_float.shape
        
        # 量化
        w_max = np.max(np.abs(w_float))
        scale = 32767.0 / w_max if w_max > 0 else 1.0
        w_int16 = np.round(w_float * scale).astype(np.int16)
        
        # 压缩 (调用新的并行处理函数)
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
    # Windows 下多进程必须在 if __name__ == '__main__': 下运行
    main()