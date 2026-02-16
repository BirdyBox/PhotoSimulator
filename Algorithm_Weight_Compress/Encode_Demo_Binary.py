import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

# ==========================================
# 1. 模式驱动自适应压缩算法 (核心逻辑)
# ==========================================
import numpy as np

def pattern_driven_bitplane_encode(
    group_int16,
    max_k=8,
    target_sqnr = 35
):
    """
    基于 bit-plane 的逐步剥离编码（理论无损）
    - group_int16: np.int16 向量
    - 每一轮显式清除已编码的 bit
    """

    # -----------------------------
    # 1. 准备：拆符号 + magnitude
    # -----------------------------
    group = group_int16.astype(np.int32)
    sign = np.sign(group)
    mag  = np.abs(group)

    mag_res = mag.copy()
    mag_rec = np.zeros_like(mag)

    total_power = np.sum(mag ** 2)
    if total_power == 0:
        return 0, 99.9

    # -----------------------------
    # 2. 逐轮剥离
    # -----------------------------
    for k in range(1, max_k + 1):

        if np.max(mag_res) == 0:
            return k - 1, 99.9

        # 找当前最高 bit-plane
        j_max = int(np.floor(np.log2(np.max(mag_res))))

        # ------------------------------------------------
        # 模式识别：判断是否可一次剥多个 bit-plane
        # 规则：只有当 group 内所有非零元素该 bit = 1
        # ------------------------------------------------
        # print(type(j_max))
        
        planes = [j_max]
        # print(planes)
        if j_max > 0:
            b_j   = (mag_res >> j_max) & 1
            b_j_1 = (mag_res >> (j_max - 1)) & 1

            # 所有非零元素在 j,j-1 位都为 1
            valid = (mag_res > 0)
            if np.all(b_j[valid] & b_j_1[valid]):
                planes.append(j_max - 1)

        if j_max > 1:
            b_j_2 = (mag_res >> (j_max - 2)) & 1
            valid = (mag_res > 0)
            if len(planes) == 2 and np.all(b_j_2[valid]):
                planes.append(j_max - 2)

        # ------------------------------------------------
        # 构造 bit mask
        # ------------------------------------------------
        mask = 0
        for j in planes:
            mask |= (1 << j)

        # ------------------------------------------------
        # 执行剥离（bit-true）
        # ------------------------------------------------
        peeled = mag_res & mask
        mag_rec += peeled
        mag_res &= ~mask

        # ------------------------------------------------
        # SQNR 计算（bit-true）
        # ------------------------------------------------
        err_power = np.sum(mag_res ** 2)
        cur_sqnr = 99.9 if err_power == 0 else \
            10 * np.log10(total_power / err_power)

        if cur_sqnr >= target_sqnr:
            return k, cur_sqnr

    return max_k, cur_sqnr

def pattern_driven_encode(group, max_k=8, target_sqnr=35):
    """
    根据位矩阵最高位模式自动选择 S 的步进
    """
    target = np.abs(group).astype(float)
    if np.sum(target) == 0:
        return 0, 99.9  # K=0

    residues = target.copy()
    reconstructed = np.zeros_like(target)
    
    for k in range(1, max_k + 1):
        # 1. 寻找当前残差中的最高有效位 (MSB)
        max_val = np.max(residues)
        if max_val < 1.0: break
        
        j_max = int(np.floor(np.log2(max_val)))
        
        # 2. 模式识别：检查组内贡献最大权重的位形态
        # 目标：识别 '11x' (双位模式) 或 '10x' (单位模式)
        top_idx = np.argmax(residues)
        val_at_top = int(residues[top_idx])
        
        bit_j = (val_at_top >> j_max) & 1
        bit_j_minus_1 = (val_at_top >> (j_max - 1)) & 1 if j_max > 0 else 0
        
        # 3. 自适应步进 S
        if bit_j == 1 and bit_j_minus_1 == 1:
            # 模式 11x: S = j-1, L=3 (11b) 可以同时干掉两个 1
            s_k = max(0, j_max - 1)
        else:
            # 模式 10x: S = j-2, L=4 (100b) 跨位覆盖 j 位
            s_k = max(0, j_max - 2)
            
        # 4. 执行量化 (MRR Level 0-4)
        levels = np.round(residues / (2.0 ** s_k))
        levels = np.clip(levels, 0, 4)
        
        comp = levels * (2.0 ** s_k)
        reconstructed += comp
        residues -= comp
        
        # 5. 计算当前质量
        err = target - reconstructed
        sig_pow = np.sum(target**2)
        err_pow = np.sum(err**2)
        cur_sqnr = 99.9 if err_pow == 0 else 10 * np.log10(sig_pow / err_pow)
        
        # 如果达到目标 SQNR，提前退出
        if cur_sqnr >= target_sqnr:
            return k, cur_sqnr
            
    return max_k, cur_sqnr



def bit_pattern_mask_encode(group, max_k=8, target_sqnr=35.0):
    """
    基于位模式屏蔽的自适应压缩算法
    实现 10x 剥离 3 位（L=4 孤立 MSB）和 11x 剥离 2 位的逻辑
    """
    target = np.abs(group).astype(np.int64) # 使用 int64 模拟硬件位操作
    if np.sum(target) == 0:
        return 0, 99.9

    residues = target.copy()
    reconstructed = np.zeros_like(target, dtype=np.int64)
    
    for k in range(1, max_k + 1):
        # 1. 找到当前残差中最大值的 MSB 位置 j
        max_val = np.max(residues)
        if max_val == 0: break
        j = int(np.floor(np.log2(max_val)))
        
        # 获取最大值在 j 和 j-1 位的情况
        # bit_j 必然是 1
        bit_j_minus_1 = (max_val >> (j - 1)) & 1 if j > 0 else 0
        
        current_levels = np.zeros_like(residues)
        s_k = 0
        
        # 2. 模式判定与 S 设定
        if bit_j_minus_1 == 1:
            # 模式 11x: 剥离 2 位窗口
            s_k = max(0, j - 1)
            for i in range(len(residues)):
                val = residues[i]
                if (val >> j) & 1:
                    current_levels[i] = 2  # 仅剥离最高位 (10b)，留下 j-1 位
                else:
                    current_levels[i] = (val >> s_k) & 1 # 剥离 j-1 位
        else:
            # 模式 10x: 剥离 3 位窗口
            s_k = max(0, j - 2)
            for i in range(len(residues)):
                val = residues[i]
                if (val >> j) & 1:
                    current_levels[i] = 4  # 核心：100b 对应 2^j，精准剥离最高位，保留低两位
                else:
                    # 最高位为 0，则将低两位无损表示掉 (0-3)
                    current_levels[i] = (val >> s_k) & 3 

        # 3. 更新重构值与残差矩阵
        current_components = current_levels << s_k
        reconstructed += current_components
        residues -= current_components
        
        # 4. 质量评估
        err = target - reconstructed
        sig_pow = np.sum(target**2)
        err_pow = np.sum(err**2)
        cur_sqnr = 99.9 if err_pow == 0 else 10 * np.log10(sig_pow / err_pow)
        
        if cur_sqnr >= target_sqnr:
            return k, cur_sqnr
            
    return max_k, cur_sqnr

def encode_4(group, max_k=8, mrr_cap=7, target_sqnr=35):
    """
    [终极版] 精准剥离 + LSB优先 + 自动位模式匹配
    """
    # 1. 预处理：确保使用 int64 进行位运算
    original = group.astype(np.float64)
    target = np.abs(original).astype(np.int64)
    signs = np.sign(original)
    
    if np.sum(target) == 0:
        return 0, 99.9

    residues = target.copy()
    reconstructed = np.zeros_like(target, dtype=np.int64)
    
    # 硬件位宽能力 (例如 4->3 bits)
    hardware_bit_width = int(np.floor(np.log2(mrr_cap))) + 1

    for k in range(1, max_k + 1):
        # A. 锚定最大值 MSB
        max_val = np.max(residues)
        if max_val == 0: break
        
        j_msb = int(np.floor(np.log2(max_val)))
        
        # B. 定义搜索窗口
        # 我们只在 MSB 附近的有限范围内搜索 S
        # 范围：[MSB - 硬件位宽 + 1, MSB]
        search_start = max(0, j_msb - hardware_bit_width + 1)
        search_end = j_msb 
        
        # C. 策略竞争 (Strategy Competition)
        best_s = -1
        min_cost = float('inf')
        best_levels = None
        
        # 关键：从 search_start (靠近 LSB) 向 search_end (靠近 MSB) 遍历
        # 这样在 cost 相同时，我们会自然保留较小的 S (因为 < 判断不更新)
        # 或者显式加入惩罚项
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
            penalty = -s_candidate * 0.8  # 这里的权重可以调整，确保在能量成本相近时优先选择小 S
            
            total_cost = energy_cost + penalty
            
            # 更新策略
            if total_cost < min_cost:
                min_cost = total_cost
                best_s = s_candidate
                best_levels = levels

        # D. 执行最佳策略
        if best_levels is None: # 兜底保护
             best_s = 0
             best_levels = np.clip(residues, 0, mrr_cap)

        current_components = best_levels << best_s
        reconstructed += current_components
        residues -= current_components # 精准减法
        
        # E. 质量检查 (SQNR)
        err = target - reconstructed
        sig_pow = np.sum(target**2)
        err_pow = np.sum(err**2)
        cur_sqnr = 99.9 if err_pow == 0 else 10 * np.log10(sig_pow / err_pow)
        
        if cur_sqnr >= target_sqnr:
            # 恢复符号并返回
            final_recon = reconstructed.astype(np.float64) * signs
            # 注意：这里的逻辑只返回 K 和 SQNR 用于统计
            # 如果需要返回权重，请修改返回值
            return k, cur_sqnr

    return max_k, cur_sqnr
# ==========================================
# 2. 效果观察器
# ==========================================

class CompressionObserver:
    def __init__(self, wdm_size=9):
        self.wdm_size = wdm_size

    def observe_layer(self, layer_weight, name):
        # 量化到 INT16
        w = layer_weight.detach().cpu().numpy()
        w_max = np.max(np.abs(w))
        scale = 32767.0 / w_max if w_max > 0 else 1.0
        w_int = np.round(w * scale).astype(np.int16).flatten()
        
        # 分组
        pad_len = (self.wdm_size - (len(w_int) % self.wdm_size)) % self.wdm_size
        groups = np.pad(w_int, (0, pad_len)).reshape(-1, self.wdm_size)
        
        # 随机采样观察
        samples = groups[np.random.choice(len(groups), min(2048, len(groups)), replace=False)]
        
        ks, sqnrs = [], []
        for g in samples:
            # k, sqnr = bit_pattern_mask_encode(g)
            # k, sqnr = pattern_driven_bitplane_encode(g)
            # k, sqnr = pattern_driven_encode(g)
            k,sqnr = encode_4(g)
            ks.append(k)
            sqnrs.append(sqnr)
            
        return {
            "Layer": name,
            "Avg_K": np.mean(ks),
            "Avg_SQNR": np.mean(sqnrs),
            "Saving": (8 - np.mean(ks)) / 8 * 100
        }

def run_observation(model_name="resnet18"):
    print(f"--- 正在观察模型: {model_name} (位模式压缩逻辑) ---")
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    else:
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        
    observer = CompressionObserver()
    results = []
    
    for name, m in tqdm(list(model.named_modules())):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            res = observer.observe_layer(m.weight, name)
            results.append(res)
            
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print(f"{model_name.upper()} 压缩效果观察简报")
    print("-"*50)
    
    print(df[["Layer", "Avg_K", "Avg_SQNR", "Saving"]].head(10).to_string())
    print("-"*50)
    #输出所有层的 Avg_K 平均值
    print(f"模型平均 Avg_K: {df['Avg_K'].mean():.4f}")
    # print(f"模型平均节省物理行: {df['Saving'].mean():.2f}%")
    print(f"模型平均重构质量: {df['Avg_SQNR'].mean():.2f} dB")
    print("="*50)

if __name__ == "__main__":
    # 可以切换为 "vgg16" 或 "alexnet"
    run_observation("resnet18")