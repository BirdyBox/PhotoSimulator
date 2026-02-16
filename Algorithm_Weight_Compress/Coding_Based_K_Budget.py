import numpy as np

# ==========================================
# 1. 全局配置参数 (用户在此处修改)
# ==========================================

# 物理限制参数
PHYSICAL_ROWS_K = 2         # K: 物理行预算 (你有几次机会去覆盖它)
MAX_MRR_LEVEL = 4           # L: MRR能级上限 (对应 0,1,2,3,4)
POSSIBLE_SHIFTS = list(range(13))  # S: 搜索范围 [0, 1, ..., 12]

# WDM 组数据 (手动设定的矩阵)
# 每一行代表一个 WDM 组，组内数据必须共享同一个 Shift (S)
# 这里模拟了三种典型情况：
TARGET_W_GROUPS = np.array([
    # Case A: 数据都比较大，且比较接近
    [1024, 1050, 2048, 2000, 1500, 1200, 4096, 4000],
    
    # Case B: 典型稀疏数据，几个大数，几个小数，中间有0
    [0, 5, 0, 4000, 0, 12, 4050, 0],
    
    # Case C: 冲突严重的数据 (既有极小值，又有极大值)
    [3, 32000, 5, 16000, 4, 8000, 2, 100]
])

# ==========================================
# 2. 核心算法实现
# ==========================================

def greedy_dense_encode(wdm_group):
    """
    对单个 WDM 组进行贪心编码
    """
    # 初始化：残差 = 原始权重 (取绝对值，因为光强只处理幅度)
    residues = np.abs(wdm_group).astype(float)
    reconstructed = np.zeros_like(residues)
    
    # 用于记录每一步的决策，方便查看
    step_logs = []

    # --- 迭代 K 次 (对应 K 行物理光路) ---
    for k in range(PHYSICAL_ROWS_K):
        best_s = 0
        best_components = np.zeros_like(residues)
        min_residual_energy = float('inf')
        
        # --- 搜索最佳 S (像移动框一样扫描) ---
        for s in POSSIBLE_SHIFTS:
            # 1. 尝试用当前 S 去解释残差
            # 公式: Level = Clip(Round(Residue / 2^s))
            scaled_vals = residues / (2 ** s)
            levels = np.round(scaled_vals)
            
            # 2. 硬件截断 (限制在 0-4 能级内)
            levels = np.clip(levels, 0, MAX_MRR_LEVEL)
            
            # 3. 计算这一行实际上能表示多少数值
            components = levels * (2 ** s)
            
            # 4. 计算如果选这个 S，剩下的误差是多少 (L2 范数)
            current_residue = residues - components
            current_energy = np.sum(current_residue ** 2)
            
            # 5. 贪心选择：保留误差最小的那个 S
            if current_energy < min_residual_energy:
                min_residual_energy = current_energy
                best_s = s
                best_components = components

        # --- 确定本行参数，更新状态 ---
        reconstructed += best_components     # 把这一行贡献的值加进去
        residues -= best_components          # 从残差里减去，剩下的留给下一行
        
        # 记录这一行的决策
        step_logs.append({
            "Row": k + 1,
            "Chosen_S": best_s,
            "Components": best_components.astype(int)
        })
        
        # 如果残差已经全为0，提前结束
        if np.all(residues == 0):
            break
            
    return reconstructed, step_logs

# ==========================================
# 3. 运行与结果展示
# ==========================================

if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"配置: K={PHYSICAL_ROWS_K} 行 | Max Level={MAX_MRR_LEVEL} | WDM Group Size={TARGET_W_GROUPS.shape[1]}")
    print(f"{'='*60}\n")

    for i, group in enumerate(TARGET_W_GROUPS):
        print(f"--- WDM Group {i+1} ---")
        print(f"原始数据: {group}")
        
        # 运行算法
        recon_group, logs = greedy_dense_encode(group)
        
        # 打印每一步的决策细节
        for log in logs:
            print(f"  > [物理行 {log['Row']}] 选择 Shift S={log['Chosen_S']:<2} | 覆盖数值: {log['Components']}")
            
        # 打印最终结果
        diff = np.abs(group) - recon_group
        mse = np.mean(diff ** 2)
        print(f"最终重构: {recon_group.astype(int)}")
        print(f"残留误差: {diff.astype(int)} (MSE: {mse:.2f})")
        print("-" * 30 + "\n")