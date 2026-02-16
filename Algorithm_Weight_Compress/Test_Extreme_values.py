import torch
import torchvision.models as models
import numpy as np
import os
from tqdm import tqdm

# ==========================================
# 工具函数：计算组内的比特分布特征
# ==========================================
def analyze_group_bits(group):
    # 1. 预处理：取绝对值，转为整数
    # 假设我们先缩放到 int16 范围来模拟定点化后的分布
    # 这里我们只关心相对比例，所以直接用 float 的 log2 也行，但转 int 更符合物理直觉
    target = np.abs(group)
    max_val = np.max(target)
    
    if max_val == 0:
        return None # 全0组，跳过
        
    # 为了统一标准，我们将每层的权重归一化到 [0, 32767] 的整数域进行分析
    # 这样比较不同层级时才公平
    # 注意：这里是按组归一化还是按层？按层归一化更符合实际量化流程。
    # 我们在主循环里做按层缩放。这里假设输入已经是 int
    
    # 过滤掉 0 值，因为 log2(0) 无意义
    non_zeros = target[target > 0]
    if len(non_zeros) == 0:
        return None
        
    # --- 指标 1: 最大值与最小值的 MSB 差 (Max-Min MSB Diff) ---
    # 物理含义：最大那个数比最小那个数高了多少个 2 的幂次
    # 如果 > 8，说明就算 8bit 量化，最小值也会变成 0
    v_max = np.max(non_zeros)
    v_min = np.min(non_zeros)
    msb_max = int(np.floor(np.log2(v_max)))
    msb_min = int(np.floor(np.log2(v_min)))
    diff_max_min = msb_max - msb_min
    
    # --- 指标 2: 最大值与中位数的 MSB 差 (Max-Median MSB Diff) ---
    # 物理含义：最大值是否是“光杆司令”？
    # 如果 Diff 很大 (比如 > 4)，说明大哥遥遥领先，大部分小弟都很小。算法需要照顾大哥。
    # 如果 Diff 很小 (比如 0 或 1)，说明大家都很均匀。
    v_median = np.median(non_zeros)
    if v_median == 0: 
        # 如果中位数是0（虽然前面过滤了0，但如果非零数很少可能发生），取非零的最小
        msb_median = 0 
        diff_max_median = msb_max # 极端情况
    else:
        msb_median = int(np.floor(np.log2(v_median)))
        diff_max_median = msb_max - msb_median

    # --- 指标 3: 整个组的有效比特位宽 (Effective Bit Span) ---
    # 物理含义：组内所有数的所有 1，分布在多宽的范围内？
    # 比如 [1000, 1]，跨度就是 log2(1000) - log2(1) = 10 bit
    # 这决定了我们需要多少个 S 才能覆盖所有信息
    # 这里的逻辑是：(组内最大的 log2) - (组内所有数做 OR 运算后的最低位 1 的位置)
    # 或者简单点：就是 MSB_max - (组内所有非零数值的最小 LSB 位置)
    # 为了简化，我们用 MSB_max - MSB_min 近似表示“跨度”
    
    return {
        'max_val': v_max,
        'msb_max': msb_max,
        'diff_max_min': diff_max_min,
        'diff_max_median': diff_max_median
    }

# ==========================================
# 主逻辑
# ==========================================
def main():
    MODEL_NAME = "vgg" # 或者 alexnet
    print(f"=== 正在分析模型: {MODEL_NAME} 的权重位宽分布 ===")
    
    model = models.vgg16(weights='DEFAULT')
    
    # 统计容器
    stats = {
        'total_groups': 0,
        'diff_max_min_hist': np.zeros(32), # 记录 Max-Min 差值的分布 (0-31)
        'diff_max_med_hist': np.zeros(32), # 记录 Max-Median 差值的分布
        'extreme_outlier_groups': 0,       # 记录极度离群的组数 (Max-Med > 6)
    }
    
    wdm_group_size = 9
    
    for name, module in tqdm(model.named_modules(), desc="Scanning Layers"):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            w = module.weight.detach().cpu().numpy()
            
            # 1. 全局量化到 int16 (模拟实际场景)
            w_abs = np.abs(w)
            w_max = np.max(w_abs)
            if w_max == 0: continue
            
            scale = 32767.0 / w_max
            w_int = np.floor(w_abs * scale).astype(np.int32) # 使用 int32 防止溢出
            
            # 2. 分组
            w_flat = w_int.flatten()
            pad_len = (wdm_group_size - (len(w_flat) % wdm_group_size)) % wdm_group_size
            padded = np.pad(w_flat, (0, pad_len), 'constant')
            groups = padded.reshape(-1, wdm_group_size)
            
            # 3. 逐组分析
            for g in groups:
                res = analyze_group_bits(g)
                if res is None: continue
                
                stats['total_groups'] += 1
                
                # 记录 Max-Min 差
                idx1 = min(res['diff_max_min'], 31)
                stats['diff_max_min_hist'][idx1] += 1
                
                # 记录 Max-Median 差
                idx2 = min(res['diff_max_median'], 31)
                stats['diff_max_med_hist'][idx2] += 1
                
                # 定义“离群组”：如果大哥比中位数大了 6 个 bit (64倍)
                # 意味着为了照顾大哥，中位数及以下的数可能都会被丢弃
                if res['diff_max_median'] >= 6:
                    stats['extreme_outlier_groups'] += 1

    # ==========================================
    # 打印报告
    # ==========================================
    total = stats['total_groups']
    print("\n" + "="*50)
    print(f"分析报告 (Total Groups: {total})")
    print("="*50)
    
    print("\n[1] 组内贫富差距 (Max MSB - Min MSB):")
    for i in range(16):
        count = stats['diff_max_min_hist'][i]
        ratio = count / total * 100
        if ratio > 0.1: # 只显示占比 > 0.1% 的
            bar = "#" * int(ratio // 2)
            print(f"  差 {i} bits: {ratio:5.2f}% {bar}")
            
    print("\n[2] 离群程度 (Max MSB - Median MSB):")
    for i in range(16):
        count = stats['diff_max_med_hist'][i]
        ratio = count / total * 100
        if ratio > 0.1:
            bar = "#" * int(ratio // 2)
            print(f"  差 {i} bits: {ratio:5.2f}% {bar}")
            
    print("\n[3] 风险评估:")
    extreme_ratio = stats['extreme_outlier_groups'] / total * 100
    print(f"  极度离群组 (Max >> Median by 6+ bits): {extreme_ratio:.2f}%")
    
    if extreme_ratio < 5:
        print("  结论: 大部分组分布均匀，当前算法策略应该是安全的。")
    else:
        print("  结论: 存在较多离群组，需要重点关注 'LSB优先' 策略是否会导致大数削顶。")

if __name__ == "__main__":
    main()