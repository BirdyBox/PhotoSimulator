import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import itertools
import copy
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor # 引入多进程

# ==========================================
# 1. 核心算法 (与之前一致)
# ==========================================
MAX_K_LIMIT = 4
MRR_LEVEL_CAP = 4
MIN_S_INTERVAL = 2
SEARCH_SPACE = list(range(13))
TARGET_SQNR_DB = 35.0
MAX_ABS_ERR_TOL = 32

# 全局缓存 (子进程会各自拥有副本)
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
    """处理单个组的函数 (将被并行调用)"""
    # 确保缓存存在
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
    """处理一层的权重 (负责多进程分发)"""
    # 填充
    pad_len = (wdm_group_size - (len(w_flat) % wdm_group_size)) % wdm_group_size
    padded = np.pad(w_flat, (0, pad_len), 'constant')
    groups = padded.reshape(-1, wdm_group_size)
    
    # --- 多进程加速核心 ---
    # 根据 CPU 核数决定 worker 数量
    max_workers = os.cpu_count()
    results = []
    
    # 使用 chunksize 减少进程间通信开销
    chunksize = max(1, len(groups) // (max_workers * 4))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # map 会保持顺序
        results = list(executor.map(compress_single_group, groups, chunksize=chunksize))
        
    recon_flat = np.concatenate(results)[:len(w_flat)]
    return recon_flat

# ==========================================
# 2. 主逻辑：压缩并保存
# ==========================================

def compress_and_save_model(output_path="resnet18_compressed.pth"):
    print(f"正在加载原始 ResNet18 模型...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 预计算缓存 (主进程)
    precompute_combos()
    
    print(f"开始压缩 (使用 {os.cpu_count()} 核并行加速)...")
    
    # 获取所有需要处理的层
    layers_to_process = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers_to_process.append((name, module))
    
    for name, module in tqdm(layers_to_process, desc="Compressing"):
        # 1. 提取参数
        w_device = module.weight.device
        w_float = module.weight.detach().cpu().numpy()
        w_shape = w_float.shape
        
        # 2. 量化
        w_max = np.max(np.abs(w_float))
        scale = 32767.0 / w_max if w_max > 0 else 1.0
        w_int16 = np.round(w_float * scale).astype(np.int16)
        
        # 3. 核心压缩 (并行加速)
        w_recon_int16_flat = process_layer_weights(w_int16.flatten(), wdm_group_size=8)
        w_recon_int16 = w_recon_int16_flat.reshape(w_shape)
        
        # 4. 反量化与写回
        w_recon_float = w_recon_int16.astype(np.float32) / scale
        module.weight.data = torch.from_numpy(w_recon_float).to(w_device)
    
    print(f"压缩完成！正在保存至 {output_path} ...")
    torch.save(model.state_dict(), output_path)
    print("保存成功。")

if __name__ == "__main__":
    # 多进程必须在 if __name__ == "__main__": 下运行
    compress_and_save_model("resnet18_bit_density_compressed.pth")