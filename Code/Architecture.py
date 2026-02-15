import math
# 假设 components.py 与此脚本在同级目录
from Components import Laser, Modulator, ADC, DAC, Memory, DigitalSlicer,AdderTree,DigitalShifter

class MRRBitSparseCore:
    """
    [Bit-Density Architecture]
    结合了 '权重空间压缩' 和 '激活值时间稀疏' 的 MRR 光计算核心。
    硬件规格: 12 行 (Spatial) x 9 列 (Spectral)
    """
    def __init__(self, config):
        self.config = config
        
        # --- 1. 硬件几何约束 ---
        self.array_rows = config.get('mrr_array_rows', 12)  # 物理行数 (Waveguides)
        self.array_cols = config.get('mrr_array_cols', 9)   # 物理列数 (Wavelengths)
        
        # --- 2. 精度参数 ---
        self.input_bit = config.get('input_precision', 16)    # e.g., INT16
        self.cell_bit = config.get('photonic_precision', 4)   # e.g., INT4
        # 理论最大切片数 (e.g., 16/4 = 4)
        self.max_slices = math.ceil(self.input_bit / self.cell_bit)

        # --- 3. 组件实例化 ---
        # 关键: Slicer 是基于此前 Synopsys DC 综合报告建模的
        self.slicer = DigitalSlicer(config) 
        self.laser = Laser(config)
        self.modulator = Modulator(config)
        self.adc = ADC(config)
        self.dac = DAC(config)
        self.adder_tree = AdderTree(config)
        self.shifter = DigitalShifter(config)
        # 预计算链路预算 (决定 Laser 功率)
        self.laser_power_mW = self._calculate_link_budget()

    def _calculate_link_budget(self):
        """计算光链路损耗，反推 Laser 功率"""
        # MRR 损耗模型: Through Loss (非谐振) + Drop Loss (谐振)
        # 每一行光穿过 array_cols 个 MRR
        loss_through = 0.05 # dB per ring
        loss_drop = 1.5     # dB
        # 考虑 worst-case: 光穿过所有环并在最后一个环下路
        total_loss_db = (self.array_cols * loss_through) + loss_drop + 6.0 # +6dB 系统裕量(耦合器等)
        
        sensitivity_dBm = self.config.get('detector_sensitivity_dBm', -20)
        required_mW = 10 ** ((sensitivity_dBm + total_loss_db) / 10)
        return required_mW

    def map_layer(self, M, N, K, weight_occupied_rows, activation_sparsity=0.0):
        """
        映射神经网络层 (GEMM: M x K * K x N)
        
        参数:
        :param M: Batch Size * H * W (输入向量数量)
        :param N: Output Channels (输出通道数)
        :param K: Input Features (输入特征数)
        :param weight_occupied_rows: [权重压缩] 压缩后，一个权重值平均占用几个 MRR 行？
                                     (例如: 原本需要4行，压缩后只需2行)
        :param activation_sparsity:  [激活稀疏] 输入切片为0的比例 (0.0 - 1.0)
        """
        
        # --- 1. 空间映射 (Spatial Mapping - Weights) ---
        # 我们的阵列有 array_rows (12) 行。
        # 如果 weight_occupied_rows = 2，那么我们一次并行计算 12 // 2 = 6 个输出通道。
        # 如果没有压缩 (weight_occupied_rows = 4)，一次只能算 12 // 4 = 3 个。
        # -> 压缩算法直接提升了吞吐量！
        
        output_channels_per_pass = self.array_rows // weight_occupied_rows
        if output_channels_per_pass < 1:
            output_channels_per_pass = 1 # 至少处理1个，此时需要多周期折叠
            
        n_passes = math.ceil(N / output_channels_per_pass)
        
        # --- 2. 波长映射 (Spectral Mapping - Reduction) ---
        # K 维度拆分到 array_cols (9) 个波长上
        k_passes = math.ceil(K / self.array_cols)
        
        # --- 3. 时间映射 (Temporal Mapping - Activations) ---
        # 动态处理：Slicer 扫描后，跳过无效切片
        # 理论切片数 = max_slices (4)
        # 有效切片数 = max_slices * (1 - sparsity)
        effective_temporal_cycles = self.max_slices * (1 - activation_sparsity)
        
        # --- 4. 总体性能统计 ---
        # 总计算周期 = (K切分) * (N切分) * (有效时间切片) * (输入向量数 M)
        total_cycles = k_passes * n_passes * effective_temporal_cycles * M
        
        # 记录映射信息用于能耗计算
        stats = {
            "M": M, "N": N, "K": K,
            "total_cycles": total_cycles,
            "output_channels_per_pass": output_channels_per_pass,
            "weight_occupied_rows": weight_occupied_rows,
            "activation_sparsity": activation_sparsity,
            "effective_temporal_cycles": effective_temporal_cycles
        }
        return stats

    def estimate_energy(self, stats):
        """
        计算能耗 (包含 Slicer 前处理代价 + 光计算代价)
        """
        M, N, K = stats["M"], stats["N"], stats["K"]
        total_cycles = stats["total_cycles"]
        sparsity = stats["activation_sparsity"]
        
        # --- A. 数字域: Slicer 前处理 ---
        # Slicer 必须扫描 *所有* 输入位，无论是否为0，才能决定是否跳过。
        # 处理的数据量 = M (向量数) * K_passes (拆分次数)
        # 假设 Slicer 也是流水线的，每次处理 array_cols 宽度的向量
        num_slicer_ops = M * math.ceil(K / self.array_cols)
        e_slicer = self.slicer.get_energy(num_slicer_ops)
        
        # --- B. 光域: 激光器 (Laser) ---
        # 激光器在所有计算周期内常开
        # 假设 1 cycle = 1 ns (1GHz)
        time_ns = total_cycles * 1.0
        e_laser = self.laser.get_power_usage(self.laser_power_mW) * time_ns
        
        # --- C. 光域: 动态调制 (Modulator + DAC) ---
        # 只有非零的切片才会触发 DAC 和 Modulator 翻转
        # 总理论操作数 (Ops) = M * N * K * max_slices
        # 实际操作数 ≈ 理论 * (1 - sparsity)
        # 注意: 这里的操作数要映射到物理器件上
        # 每次有效的 Cell 更新需要: 1个 DAC 动作 + 1个 Modulator 动作
        
        # 估算实际触发了多少次“乘加”
        total_mac_ops = M * N * K * self.max_slices * (1 - sparsity)
        
        # 映射到器件层级:
        # 每个 MAC 对应 1个 cell_bit (4-bit) 的调制
        e_dac = self.dac.get_energy(total_mac_ops)
        e_mod = self.modulator.get_energy(total_mac_ops * self.cell_bit)
        
        # --- D. 光域: 读出 (ADC) ---
        # ADC 的触发次数取决于输出的生成频率
        # 每一行(Waveguide)产生部分和，都需要 ADC 读取？
        # 通常是在累加完成后读取，或者每周期读取并在数字域累加。
        # 假设: 每周期都需要 ADC 读取 (Digital Accumulation)
        # 激活的 ADC 数量 = 激活的行数 * 周期数
        active_rows = stats["output_channels_per_pass"] * stats["weight_occupied_rows"]
        # 注意：如果 weight_occupied_rows=2，说明2个行算1个值，这2行都需要ADC读出然后数字相加
        
        total_adc_reads = total_cycles * active_rows
        e_adc = self.adc.get_energy(total_adc_reads)
        
        # [新增] 数字后处理 (Digital Post-Processing)
        # 对于每个输出值 (Total Output Elements = M * N)
        # 我们需要合并 stats['effective_temporal_cycles'] 个切片结果
    
        # 1. Shift: 除了最低位切片，其他都需要移位
        # 假设平均每个值由 4 个切片组成
        num_shifts = stats["M"] * stats["N"] * (self.max_slices - 1)
        e_shift = self.shifter.get_energy(num_shifts, bit_width=32) # 累加器通常位宽较大
        
        # 2. Add: 将移位后的结果累加
        # 相当于 N 个切片做归约，需要 N-1 次加法
        # 这里我们复用 AdderTree 模型，inputs = effective_temporal_cycles
        # 但由于是时序累加，其实就是简单的累加器，也可以看作 Input=2 的 AdderTree 被调用多次
        num_adds = stats["M"] * stats["N"] * (self.max_slices - 1)
        e_add = self.adder_tree.get_energy(num_inputs=2, num_trees=num_adds)

        total_energy = e_slicer + e_laser + e_dac + e_mod + e_adc + e_shift + e_add
        
        return total_energy, {
            "slicer": e_slicer,
            "laser": e_laser,
            "dynamic": e_dac + e_mod + e_adc,
            "total": total_energy
        }

# --- 仿真主程序 (Main Simulation) ---功能验证性测试
if __name__ == "__main__":
    # 1. 定义配置 
    config = {
        # 硬件规格
        'mrr_array_rows': 12,
        'mrr_array_cols': 9,
        'frequency': 1e9,          # 1 GHz 
        
        # 精度参数
        'input_precision': 16,     # 16-bit Input
        'photonic_precision': 4,   # 4-bit Cell
        
        # 器件参数 (Slicer 数据来自 power_report.txt)
        'slicer_energy_per_cycle_pJ': 30.2, 
        'laser_wall_plug_eff': 0.2,
        'detector_sensitivity_dBm': -22,
        'dac_energy_per_sample_pJ': 0.5,
        'adc_energy_per_sample_pJ': 1.0,
    }
    
    core = MRRBitSparseCore(config)
    
    print("=== Bit-Density Architecture Simulation ===")
    print(f"Core Size: {core.array_rows}x{core.array_cols}")
    print(f"Slicer Energy: {config['slicer_energy_per_cycle_pJ']} pJ/op")
    
    # 2. 定义工作负载 (例如: ResNet50 的一个卷积层)
    # Conv2d: 64 -> 128, 3x3 kernel, 56x56 image
    M = 56 * 56 # 3136 vectors
    N = 128     # Output Channels
    K = 64 * 3 * 3 # 576 Input Features
    
    # 3. 对比实验
    
    # [Baseline]: 无压缩，无稀疏
    # 16-bit 权重占用 4 行 (12行只能并算 3 个通道)
    # 激活稀疏度 0%
    stats_base = core.map_layer(M, N, K, weight_occupied_rows=4, activation_sparsity=0.0)
    e_base, d_base = core.estimate_energy(stats_base)
    
    # [Ours]: 权重压缩 + 激活稀疏
    # 假设算法将权重压缩到平均占用 2 行 (吞吐量翻倍!)
    # 假设 Slicer 发现 60% 的切片是 0
    stats_ours = core.map_layer(M, N, K, weight_occupied_rows=2, activation_sparsity=0.6)
    e_ours, d_ours = core.estimate_energy(stats_ours)
    
    print("\n--- Results Comparison ---")
    print(f"[Baseline] Latency: {stats_base['total_cycles']} cycles | Energy: {e_base/1e6:.2f} uJ")
    print(f"[Ours]     Latency: {stats_ours['total_cycles']} cycles | Energy: {e_ours/1e6:.2f} uJ")
    
    speedup = stats_base['total_cycles'] / stats_ours['total_cycles']
    energy_saving = (e_base - e_ours) / e_base * 100
    
    print(f"\n>>> Performance Gain <<<")
    print(f"Speedup: {speedup:.2f}x (来源: 空间并行度提升 + 跳过无效时间切片)")
    print(f"Energy Saving: {energy_saving:.2f}%")
    
    # 验证 Slicer 开销是否值得
    slicer_overhead = d_ours['slicer'] / e_ours * 100
    print(f"Slicer Overhead: {slicer_overhead:.2f}% (如果过高，说明稀疏收益被抵消)")