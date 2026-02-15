import math

class ComponentBase:
    """所有硬件组件的基类"""
    def __init__(self, config):
        self.config = config
        # 从配置中读取工作频率，默认为5GHz
        self.freq = config.get('frequency', 5e9) 

class Laser(ComponentBase):
    """
    激光器模型
    参考逻辑: Lightening-Transformer/hardware/photonic_core_base.py
    功能: 计算产生特定光功率所需的电功率 (Wall-Plug Efficiency)
    """
    def __init__(self, config):
        super().__init__(config)
        # 插墙效率 (例如 0.2 表示 20% 的电能转化为光能)
        self.wall_plug_efficiency = config.get('laser_wall_plug_eff', 0.2)
        # 激光器自身的静态功耗 (如果有)
        self.static_power = config.get('laser_static_power_mW', 0)

    def get_power_usage(self, required_optical_power_mW):
        """
        计算电功耗
        :param required_optical_power_mW: 链路预算计算出的所需光功率
        :return: 电功率 (mW)
        """
        electrical_power = (required_optical_power_mW / self.wall_plug_efficiency) + self.static_power
        return electrical_power

class Modulator(ComponentBase):
    """
    光调制器模型 (支持 MRR 或 MZI)
    参考逻辑: Lightening-Transformer 区分了 MRR (需加热) 和 MZI (面积大)
    """
    def __init__(self, config):
        super().__init__(config)
        self.type = config.get('modulator_type', 'MRR') # MRR or MZI
        
        # 动态调制能耗 (fJ/bit)
        # 参考 DAC.py 或相关文献，通常是 CV^2/4 级别
        self.energy_per_bit = config.get('modulator_energy_per_bit_fJ', 10) 
        
        # MRR 特有的加热功耗 (用于波长对准)
        self.heater_power_mW = config.get('mrr_heater_power_mW', 0) if self.type == 'MRR' else 0

    def get_energy(self, num_bits, active_time_ns=None):
        """
        计算总能耗 (动态 + 静态)
        :param num_bits: 传输的总比特数
        :param active_time_ns: 激活时间，用于计算静态加热功耗
        :return: 能量 (pJ)
        """
        # 动态能耗: bits * fJ/bit -> pJ (1 fJ = 1e-3 pJ)
        dynamic_energy = num_bits * self.energy_per_bit * 1e-3
        
        # 静态能耗: Power * Time
        static_energy = 0
        if self.type == 'MRR' and active_time_ns is not None:
            # mW * ns = pJ
            static_energy = self.heater_power_mW * active_time_ns
            
        return dynamic_energy + static_energy

class ADC(ComponentBase):
    """
    模数转换器模型
    参考逻辑: Lightening-Transformer/hardware/ADC.py
    """
    def __init__(self, config):
        super().__init__(config)
        self.precision = config.get('precision', 8) # bits
        self.sample_rate = self.freq # 假设采样率等于系统频率
        
        # 方式1: 直接给定每采样能耗 (参考 ISAAC 架构: ~1.2 fJ/conversion step)
        # Lightening-Transformer 中 ADC.py 使用了特定的字典查找
        self.energy_per_sample_pJ = config.get('adc_energy_per_sample_pJ', 0.5) 
        
        # 方式2: 基于 Walden Figure of Merit (FOM) 计算: P = FOM * 2^ENOB * Fs
        # 如果 config 中提供了 FOM，则覆盖上面的静态值
        if 'adc_fom_fJ_per_step' in config:
            fom = config['adc_fom_fJ_per_step']
            # Power (mW) = FOM (fJ) * 2^bits * Fs (GHz) / 1000
            # Energy per sample (pJ) = Power / Fs = FOM * 2^bits * 1e-3
            self.energy_per_sample_pJ = fom * (2**self.precision) * 1e-3

    def get_energy(self, num_samples):
        """
        :param num_samples: 需要转换的样本数量
        :return: 能量 (pJ)
        """
        return num_samples * self.energy_per_sample_pJ

class DAC(ComponentBase):
    """
    数模转换器模型
    参考逻辑: Lightening-Transformer/hardware/DAC.py
    """
    def __init__(self, config):
        super().__init__(config)
        self.precision = config.get('precision', 8)
        self.energy_per_sample_pJ = config.get('dac_energy_per_sample_pJ', 0.1)

    def get_energy(self, num_samples):
        return num_samples * self.energy_per_sample_pJ

class Memory(ComponentBase):
    """
    片上存储模型 (SRAM/Global Buffer)
    参考逻辑: Lightening-Transformer/hardware/SRAM.py
    """
    def __init__(self, config):
        super().__init__(config)
        # 单位: pJ/bit
        self.read_energy_per_bit = config.get('memory_read_energy_pJ_bit', 0.05)
        self.write_energy_per_bit = config.get('memory_write_energy_pJ_bit', 0.06)
        # 静态漏功耗 (mW/MB)
        self.leakage_power_per_MB = config.get('memory_leakage_power_mW_MB', 10)
        self.size_MB = config.get('memory_size_MB', 2)

    def get_read_energy(self, num_bits):
        return num_bits * self.read_energy_per_bit

    def get_write_energy(self, num_bits):
        return num_bits * self.write_energy_per_bit
        
    def get_leakage_energy(self, time_ns):
        # mW * ns = pJ
        return self.leakage_power_per_MB * self.size_MB * time_ns
    
class DigitalSlicer(ComponentBase):
    """
    数字位切片器 (Bit Slicer)
    基于 Synopsys Design Compiler 综合报告建模 (65nm 工艺)
    数据来源: power_report.txt, area_report.txt
    """
    def __init__(self, config):
        super().__init__(config)
        # 基础参数从报告中提取
        # 报告基于 100MHz (10ns)，总动态功耗 3.0207 mW
        # 计算得单周期能耗: 3.0207 mW * 10 ns = 30.207 pJ
        self.base_energy_per_cycle_pJ = config.get('slicer_energy_per_cycle_pJ', 30.21)
        
        # 面积 (um^2)
        self.area_um2 = config.get('slicer_area_um2', 36587.88)
        
        # 漏功耗 (mW), 报告值为 365.19 nW = 0.000365 mW
        self.leakage_power_mW = config.get('slicer_leakage_power_mW', 3.65e-4)
        
        # 吞吐量假设：该模块在一个周期内能处理多少个输入向量？
        # 假设 dispatch_system_top 是为整个 Tile (例如 64x64) 服务的
        self.process_width = config.get('core_width', 64)

    def get_energy(self, num_vectors):
        """
        计算切片操作的能耗
        :param num_vectors: 需要处理的输入向量数量 (对应矩阵的行数或 Batch)
        """
        # 假设硬件是流水线的，每周期处理 1 个向量 (或一组)
        # 动态能耗 = 向量数 * 单周期能耗
        dynamic_energy = num_vectors * self.base_energy_per_cycle_pJ
        return dynamic_energy

    def get_area(self):
        return self.area_um2
    
class DigitalShifter(ComponentBase):
    """
    电子移位器 (Barrel Shifter / Shift Register)
    用于 Bit-Slicing 架构中，将低精度计算结果左移恢复权重 (e.g., x << 4)。
    
    参数参考:
    - 惯例数值: 移位器通常比加法器简单。
    - 假设 32-bit 移位器能耗约为 32-bit 加法器的 20% - 30%。
    """
    def __init__(self, config):
        super().__init__(config)
        # 默认每 bit 移位的能耗 (fJ)
        # 参考 65nm/45nm 数字逻辑: 简单逻辑翻转 ~5-10 fJ/bit
        self.energy_per_bit_fJ = config.get('shifter_energy_per_bit_fJ', 5.0)

    def get_energy(self, num_values, bit_width=32):
        """
        :param num_values: 需要移位的数值个数
        :param bit_width: 数据的位宽
        """
        total_bits = num_values * bit_width
        # fJ -> pJ
        return total_bits * self.energy_per_bit_fJ / 1000.0

class ElectronicDemux(ComponentBase):
    """
    电子解复用器 (1-to-N Demux)
    用于将串行数据流分发到并行的 MRR 行或缓存中。
    将并行的的MRR行,在加法树上还原到对应的位置
    """
    def __init__(self, config):
        super().__init__(config)
        # 每 bit 通过 Demux 的能耗
        # 这是一个非常轻量级的逻辑，主要由传输门电容充放电决定
        self.energy_per_bit_fJ = config.get('demux_energy_per_bit_fJ', 2.0)

    def get_energy(self, num_bits_total):
        return num_bits_total * self.energy_per_bit_fJ / 1000.0
    
class AdderTree(ComponentBase):
    """
    数字加法树 (Adder Tree)
    用于将光核输出的多个部分和 (Partial Sums) 在数字域进行累加。
    
    参数参考: 
    - Lightening-Transformer (simulator_attn.py): 
      "self.adder_power = 0.2 / 4.39  # follow tech node scaling law to 14nm # mW @ ISAAC"
      这对应于 5GHz 下的功率。
      单次加法能耗 (pJ) = Power (mW) / Freq (GHz)
    """
    def __init__(self, config):
        super().__init__(config)
        # 基础功率 (mW) @ Reference Frequency
        # Lightening-Transformer 使用 ~0.045 mW (32-bit INT adder in 14nm estimate)
        self.base_power_mW = config.get('adder_power_mW', 0.045) 
        self.ref_freq_GHz = config.get('adder_ref_freq_GHz', 5.0) # 基准频率
        
        # 计算单次加法操作的能耗 (pJ)
        # Energy = Power / Frequency
        self.energy_per_op_pJ = self.base_power_mW / self.ref_freq_GHz

    def get_energy(self, num_inputs, num_trees=1):
        """
        计算加法树能耗
        :param num_inputs: 加法树的输入数量 (叶子节点数)
        :param num_trees: 并行有多少个这样的树 (例如并行处理多少个输出通道)
        """
        if num_inputs <= 1:
            return 0
            
        # N 个输入的树包含 N-1 个加法器
        num_adds_per_tree = num_inputs - 1
        total_ops = num_adds_per_tree * num_trees
        
        return total_ops * self.energy_per_op_pJ

# ---------------------------------------------------------
# 简单的测试代码 (Self-Test)
# ---------------------------------------------------------
if __name__ == "__main__":
    # 模拟从 yaml 读取的配置
    dummy_config = {
        'frequency': 5e9,                # 5 GHz
        'precision': 8,                  # 8-bit system
        'laser_wall_plug_eff': 0.25,     # 25% WPE
        'modulator_type': 'MRR',
        'mrr_heater_power_mW': 5.0,      # MRR 热调功耗
        'adc_energy_per_sample_pJ': 1.2, # ADC 单次转换能耗
        'memory_size_MB': 4,
        'adder_power_mW': 0.045, # Lightening-Transformer 参数
        'shifter_energy_per_bit_fJ': 5.0,
        'demux_energy_per_bit_fJ': 2.0
    }
    

    print("=== 初始化组件 ===")
    laser = Laser(dummy_config)
    adc = ADC(dummy_config)
    mem = Memory(dummy_config)
    mod = Modulator(dummy_config)
    slicer = DigitalSlicer(dummy_config)
    # 1. Adder Tree 测试
    # 假设我们有 9 个 MRR 波长产生 9 个部分和，需要 1 个树来规约
    adder = AdderTree(dummy_config)
    ops = 9 - 1
    e_add = adder.get_energy(num_inputs=9, num_trees=1)
    print(f"Adder Tree (9 inputs): {e_add:.4f} pJ (Single Adder: {adder.energy_per_op_pJ:.4f} pJ)")
    
    # 2. Shifter 测试
    # 对 1 个 32-bit 数进行移位
    shifter = DigitalShifter(dummy_config)
    e_shift = shifter.get_energy(num_values=1, bit_width=32)
    print(f"Shifter (1x32-bit):    {e_shift:.4f} pJ")
    
    # 3. Demux 测试
    demux = ElectronicDemux(dummy_config)
    e_demux = demux.get_energy(32)
    print(f"Demux (32-bit):        {e_demux:.4f} pJ")
    print(f"1. Laser Power for 10mW optical: {laser.get_power_usage(10):.2f} mW (Expect 40.0)")
    print(f"2. ADC Energy for 1M samples: {adc.get_energy(1e6):.2f} pJ")
    print(f"3. Memory Read Energy for 1MB data: {mem.get_read_energy(8 * 1024 * 1024):.2f} pJ")
    print(f"4. Modulator Static Energy (10ns): {mod.get_energy(0, active_time_ns=10):.2f} pJ")
    print(f"5. DigtialSlicer Energy:{slicer.get_energy(10):.2f} pJ")