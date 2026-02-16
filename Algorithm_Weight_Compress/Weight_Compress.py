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

# ==========================================
# 3. 主逻辑
# ==========================================
def main():
    # --- 配置区域 ---
    MODEL_NAME = "vgg16"  # <=== 修改这里！可选: 'alexnet', 'vgg16', 'mobilenet_v2', 'resnet50'
    OUTPUT_FILE = f"{MODEL_NAME}_compressed.pth"
    
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
        w_recon_int16_flat = process_layer_weights(w_int16.flatten(), wdm_group_size=8)
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