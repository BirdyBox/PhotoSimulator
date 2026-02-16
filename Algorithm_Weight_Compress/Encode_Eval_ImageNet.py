import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import copy
import itertools
from tqdm import tqdm
import os
from PIL import Image
import os

class FlatImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, val_txt_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        
        # 解析 val.txt
        # 假设每一行格式为: "图片文件名 类别索引" (例如: ILSVRC2012_val_00000001.JPEG 65)
        with open(val_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    label = int(parts[1])
                    self.samples.append((filename, label))
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)
        
        # 必须转换为 RGB，防止部分黑白图片导致 Tensor 维度错误
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
# ==========================================
# 1. 核心压缩算法 (必须修改为返回重构权重)
# ==========================================

# 全局配置
MAX_K_LIMIT = 4
MRR_LEVEL_CAP = 4
MIN_S_INTERVAL = 2
SEARCH_SPACE = list(range(13))
TARGET_SQNR_DB = 35.0
MAX_ABS_ERR_TOL = 32

# 预计算缓存
VALID_COMBOS_CACHE = {}

def precompute_combos():
    """预计算 S 组合"""
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
    """使用给定 S 组合进行拟合，返回重构数据"""
    residues = residues_in.copy()
    recon = np.zeros_like(residues)
    for s in sorted(s_combo, reverse=True):
        levels = np.floor(residues / (2.0 ** s))
        levels = np.clip(levels, 0, MRR_LEVEL_CAP)
        comp = levels * (2.0 ** s)
        recon += comp
        residues -= comp
    return recon

def get_compressed_weights(group):
    """
    核心修改：此函数现在直接返回 '最佳重构权重' 而不是 metrics
    """
    target = np.abs(group).astype(float)
    if np.sum(target) == 0:
        return np.zeros_like(group) # 全0直接返回

    best_recon = np.zeros_like(target)
    
    # 动态 K 搜索
    for k in range(1, MAX_K_LIMIT + 1):
        k_best_sqnr = -1
        k_best_recon = None
        
        for combo in VALID_COMBOS_CACHE[k]:
            recon = fit_with_s_combo(target, combo)
            
            # 计算 SQNR
            err = target - recon
            err_pow = np.sum(err**2)
            if err_pow == 0: cur_sqnr = 99.9
            else:
                sig_pow = np.sum(target**2)
                cur_sqnr = 10 * np.log10(sig_pow / err_pow)
            
            if cur_sqnr > k_best_sqnr:
                k_best_sqnr = cur_sqnr
                k_best_recon = recon
        
        # 检查是否达标
        max_err = np.max(np.abs(target - k_best_recon))
        if k_best_sqnr >= TARGET_SQNR_DB or max_err <= MAX_ABS_ERR_TOL:
            # 还原符号（Bit-Density通常处理幅度，符号需额外处理，这里假设符号保留原样）
            # 简单处理：重构出的幅度 * 原始符号
            return k_best_recon * np.sign(group)
            
    return k_best_recon * np.sign(group)

# ==========================================
# 2. 模型注入器 (In-Memory Hot-Swap)
# ==========================================

def inject_compressed_weights(model, wdm_group_size=8):
    """
    遍历模型，将权重替换为压缩后的版本
    """
    print("正在执行模型权重热替换 (Compression Injection)...")
    precompute_combos() # 确保算法缓存已初始化
    
    # 深拷贝模型以免影响原版（可选）
    compressed_model = copy.deepcopy(model)
    
    for name, module in tqdm(compressed_model.named_modules(), desc="Compressing Layers"):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 1. 提取与量化
            device = module.weight.device
            w_float = module.weight.detach().cpu().numpy()
            w_max = np.max(np.abs(w_float))
            scale = 32767.0 / w_max if w_max > 0 else 1.0
            
            w_int16 = np.round(w_float * scale).astype(np.int16)
            
            # 2. 展平与分组
            original_shape = w_int16.shape
            flat = w_int16.flatten()
            pad_len = (wdm_group_size - (len(flat) % wdm_group_size)) % wdm_group_size
            padded = np.pad(flat, (0, pad_len), 'constant')
            groups = padded.reshape(-1, wdm_group_size)
            
            # 3. 并行/循环处理所有组 (这里为了简单用循环，慢但稳)
            # 实际部署建议用多进程池加速
            recon_groups = []
            for group in groups:
                recon_g = get_compressed_weights(group)
                recon_groups.append(recon_g)
            
            # 4. 还原形状与反量化
            recon_flat = np.concatenate(recon_groups)[:len(flat)]
            recon_int16 = recon_flat.reshape(original_shape)
            
            # 反量化回 float32
            w_recon_float = recon_int16.astype(np.float32) / scale
            
            # 5. 写入模型
            module.weight.data = torch.from_numpy(w_recon_float).to(device)
            
    return compressed_model

# ==========================================
# 3. 验证主程序
# ==========================================

def validate(model, val_loader, device):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for images, target in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            target = target.to(device)
            
            # compute output
            output = model(images)
            
            # measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
    return top1.avg, top5.avg

class AverageMeter(object):
    """计算平均值"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """计算 Top-K 准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():
    # --- 配置 ---
    BATCH_SIZE = 256
    WORKERS = 8
    # VAL_DIR = "D:\Code\Bit Density\ImageNet100\val.txt"  # <--- 请修改这里
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # if not os.path.exists(VAL_DIR):
    #     print(f"错误: 找不到路径 {VAL_DIR}，请配置 ImageNet 验证集路径。")
    #     return

    # 1. 准备数据
    print(f"正在加载 ImageNet 数据 ({DEVICE})...")
    
    # 你的 val.txt 路径
    VAL_TXT_PATH = r"D:\Code\Bit Density\\Dataset\\val.txt" # <--- 请修改这里: 指向你的 txt 文件
    IMG_DIR = r"D:\Code\Bit Density\\Dataset"     # <--- 请修改这里: 指向存放所有图片的文件夹
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    val_dataset = FlatImageNetDataset(
        img_dir=IMG_DIR, 
        val_txt_path=VAL_TXT_PATH,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True
    )

    # 2. 加载原始模型
    print("加载 ResNet18 基准模型...")
    model = models.resnet18(pretrained=True).to(DEVICE)
    
    # 3. 跑 Baseline
    print("\n>>> 开始 Baseline 测试 <<<")
    acc1_base, acc5_base = validate(model, val_loader, DEVICE)
    print(f"Baseline Result: Top1={acc1_base:.2f}%, Top5={acc5_base:.2f}%")
    
    # 4. 执行压缩
    print("\n>>> 执行 Bit-Density 压缩 (此过程较慢，请耐心等待) <<<")
    # 注意：这里我们不对模型进行采样，而是全量压缩每一层
    compressed_model = inject_compressed_weights(model, wdm_group_size=8)
    compressed_model = compressed_model.to(DEVICE)
    
    # 5. 跑 Compressed Test
    print("\n>>> 开始压缩模型测试 <<<")
    acc1_comp, acc5_comp = validate(compressed_model, val_loader, DEVICE)
    
    # 6. 输出对比报告
    print("\n" + "="*50)
    print("最终对比报告")
    print("="*50)
    print(f"{'Model':<15} | {'Top-1 Acc':<10} | {'Top-5 Acc':<10} | {'Drop (Top-1)':<10}")
    print(f"{'-'*15}-|-{'-'*10}-|-{'-'*10}-|-{'-'*10}")
    print(f"{'Original':<15} | {acc1_base.item():<10.2f} | {acc5_base.item():<10.2f} | -")
    print(f"{'Compressed':<15} | {acc1_comp.item():<10.2f} | {acc5_comp.item():<10.2f} | {(acc1_base - acc1_comp).item():.2f}")
    print("="*50)

if __name__ == "__main__":
    main()