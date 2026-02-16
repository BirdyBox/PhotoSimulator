import numpy as np
import itertools

# ==========================================
# 1. 全局配置与数据
# ==========================================

# 精度指标配置
TARGET_SQNR_DB = 40.0       # 目标信噪比 (dB)，达到这个值就停止增加 K
MAX_ABS_ERROR_TOL = 32      # 允许的最大绝对误差容忍度 (例如忽略低位噪声)

# 硬件与算法约束
MAX_K_LIMIT = 4             # 物理行上限
MRR_LEVEL_CAP = 4           # MRR 最大能级 (100)
MIN_S_INTERVAL = 2          # S 之间的最小间距 (排斥域，防止重叠内卷)
SEARCH_SPACE = list(range(13)) # S 的候选范围 [0...12]

# 测试数据 (WDM 组)
# 包含稀疏数据、密集数据、大跨度数据
TARGET_DATA = np.array([
    [1024, 1050, 2048, 2000, 1500, 1200, 4096, 4000], # 密集大数
    [0, 5, 0, 4000, 0, 12, 4050, 0],                  # 稀疏大跨度
    [32767, 100, 50, 20, 10, 5, 1, 0],                # 极端跨度
    [5, 3, 2, 1, 0, 0, 0, 0]                          # 极小值
])

# ==========================================
# 2. 辅助函数：精度评估
# ==========================================

def evaluate_accuracy(original, reconstructed):
    """
    计算 SQNR 和 最大绝对误差
    """
    error = original - reconstructed
    max_abs_err = np.max(np.abs(error))
    
    signal_power = np.sum(original.astype(float) ** 2)
    error_power = np.sum(error.astype(float) ** 2)
    
    if error_power == 0:
        sqnr = float('inf')
    elif signal_power == 0:
        sqnr = 0.0
    else:
        sqnr = 10 * np.log10(signal_power / error_power)
        
    return sqnr, max_abs_err

# ==========================================
# 3. 核心拟合逻辑：保守填充 (Conservative Fit)
# ==========================================

def fit_weights_with_fixed_s_combo(group, s_combo):
    """
    给定一组固定的 S (例如 [8, 4, 0])，计算它们能构成的最佳重构值。
    策略：从大 S 到小 S 依次向下取整填充 (Floor)，避免过充。
    """
    residues = group.astype(float) # 必须用副本
    reconstructed = np.zeros_like(residues)
    
    # 将 S 从大到小排序，优先满足高位
    # 物理意义：先用大框把大轮廓框住，再用小框修细节
    sorted_s = sorted(s_combo, reverse=True)
    
    components_log = []
    
    for s in sorted_s:
        # 1. 计算能级 (Floor 模式，杜绝 overshoot)
        # 即使 residue 是 1.9 * 2^s，也只取 1，绝不取 2
        levels = np.floor(residues / (2 ** s))
        
        # 2. 硬件截断
        levels = np.clip(levels, 0, MRR_LEVEL_CAP)
        
        # 3. 计算分量
        comp = levels * (2 ** s)
        
        # 4. 更新状态
        reconstructed += comp
        residues -= comp
        
        components_log.append((s, comp))
        
    return reconstructed, components_log

# ==========================================
# 4. 主算法：正交组合搜索
# ==========================================

def optimal_orthogonal_encoding(wdm_group):
    # 预处理：取绝对值
    target = np.abs(wdm_group)
    
    # 如果全 0 直接返回
    if np.sum(target) == 0:
        return np.zeros_like(target), 0, "All Zero"

    best_recon = np.zeros_like(target)
    final_s_combo = []
    
    # --- 动态增加 K (1 -> MAX) ---
    for k in range(1, MAX_K_LIMIT + 1):
        
        # 1. 生成所有合法的 S 组合 (S_diff >= MIN_INTERVAL)
        # itertools.combinations 生成的是有序的，只需检查相邻元素
        valid_combos = []
        for combo in itertools.combinations(SEARCH_SPACE, k):
            # 检查间距 (向量化检查或简单循环)
            # 例如 combo = (0, 3, 6)，diffs = [3, 3] >= 2 -> Valid
            is_valid = True
            for i in range(len(combo) - 1):
                if combo[i+1] - combo[i] < MIN_S_INTERVAL:
                    is_valid = False
                    break
            if is_valid:
                valid_combos.append(combo)
        
        # 如果没有合法的组合 (比如 K 太大，S 空间不够塞)，跳出
        if not valid_combos:
            break

        # 2. 遍历所有合法组合，寻找当前 K 下的最优解
        k_best_sqnr = -1
        k_best_recon = None
        k_best_combo = None
        
        for combo in valid_combos:
            recon, _ = fit_weights_with_fixed_s_combo(target, combo)
            sqnr, _ = evaluate_accuracy(target, recon)
            
            if sqnr > k_best_sqnr:
                k_best_sqnr = sqnr
                k_best_recon = recon
                k_best_combo = combo
        
        # 3. 检查是否满足停止条件 (ACC Check)
        # 获取当前最佳结果的指标
        _, max_err = evaluate_accuracy(target, k_best_recon)
        
        # 记录这一轮的成果
        best_recon = k_best_recon
        final_s_combo = k_best_combo
        
        # 判定：如果 SQNR 达标 或者 误差在容忍范围内，停止增加 K
        if k_best_sqnr >= TARGET_SQNR_DB or max_err <= MAX_ABS_ERROR_TOL:
            return best_recon, k, final_s_combo
            
    # 如果跑满 K 还没满足，返回最后一次的结果
    return best_recon, MAX_K_LIMIT, final_s_combo

# ==========================================
# 5. 运行脚本
# ==========================================

if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"策略: 正交组合搜索 | 最小间距: {MIN_S_INTERVAL} | 停止条件: SQNR>{TARGET_SQNR_DB}dB 或 Err<{MAX_ABS_ERROR_TOL}")
    print(f"{'='*60}\n")
    
    for i, group in enumerate(TARGET_DATA):
        print(f"--- WDM Group {i+1} ---")
        print(f"原始数据: {group}")
        
        # 运行优化
        recon, k_used, s_combo = optimal_orthogonal_encoding(group)
        
        # 计算最终指标
        sqnr, max_err = evaluate_accuracy(np.abs(group), recon)
        
        # 输出
        print(f"优化结果: 使用 K={k_used} 行 | 组合 S={s_combo}")
        print(f"重构数据: {recon.astype(int)}")
        print(f"误差指标: SQNR={sqnr:.2f} dB | MaxAbsErr={max_err:.0f}")
        
        # 简单的残差可视化
        residue = np.abs(group) - recon
        print(f"残留误差: {residue.astype(int)}")
        
        if sqnr >= TARGET_SQNR_DB:
            print(">> 状态: 完美达标 (High SQNR)")
        elif max_err <= MAX_ABS_ERROR_TOL:
            print(">> 状态: 勉强达标 (误差在容忍范围内)")
        else:
            print(">> 状态: 未达标 (需更多 K 或 S 策略调整)")
            
        print("-" * 30 + "\n")