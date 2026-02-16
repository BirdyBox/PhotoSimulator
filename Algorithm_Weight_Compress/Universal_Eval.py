import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# ==========================================
# 1. 核心分析器：位切片稀疏性统计 (保持不变)
# ==========================================
class BitSparsityProfiler:
    def __init__(self, bit_depth=16):
        self.bit_depth = bit_depth
        self.granularities = [16, 8, 4, 2, 1] 
        self.stats = {}
        self.layer_types = {}
        self.total_activations_count = 0

    def _update_stats(self, name, tensor_int16):
        flat_t = tensor_int16.view(-1)
        num_elements = flat_t.numel()
        self.total_activations_count += num_elements
        
        if name not in self.stats:
            self.stats[name] = {g: {'total': 0, 'zeros': 0} for g in self.granularities}

        for width in self.granularities:
            slices_per_val = self.bit_depth // width
            total_slices = num_elements * slices_per_val
            zero_slices_count = 0
            mask = (1 << width) - 1
            
            for shift in range(0, self.bit_depth, width):
                if shift == 0:
                    sliced = flat_t & mask
                else:
                    sliced = (flat_t >> shift) & mask
                zero_slices_count += torch.sum(sliced == 0).item()

            self.stats[name][width]['total'] += total_slices
            self.stats[name][width]['zeros'] += zero_slices_count

    def hook_fn(self, name):
        def forward_hook(module, input, output):
            with torch.no_grad():
                val_float = output.detach()
                max_val = val_float.abs().max()
                if max_val > 0:
                    scale = 32767.0 / max_val
                    val_int16 = (val_float * scale).round().to(torch.int16)
                else:
                    val_int16 = torch.zeros_like(val_float, dtype=torch.int16)
                self._update_stats(name, val_int16)
        return forward_hook

    def report(self):
        data = []
        for name, layer_stats in self.stats.items():
            row = {'Layer': name, 'Type': self.layer_types.get(name, 'Unknown')}
            for g in self.granularities:
                t = layer_stats[g]['total']
                z = layer_stats[g]['zeros']
                sparsity = (z / t * 100) if t > 0 else 0
                row[f'Sparsity_{g}bit'] = sparsity
            data.append(row)
        return pd.DataFrame(data)

# ==========================================
# 2. 自定义 Dataset (来自 Universal_Eval.py)
# ==========================================
class FlatImageNetDataset(Dataset):
    def __init__(self, img_dir, val_txt_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        
        if not os.path.exists(val_txt_path):
             raise FileNotFoundError(f"验证列表文件未找到: {val_txt_path}")
             
        with open(val_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # 假设 txt label 是 0-999 (标准)
                    self.samples.append((parts[0], int(parts[1])))
        
        print(f"已加载 {len(self.samples)} 张图片路径。")

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        path = os.path.join(self.img_dir, filename)
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"无法读取图片: {path}, Error: {e}")
            # 返回一个全黑图片防止崩溃，或者抛出异常
            img = Image.new('RGB', (224, 224))
            
        if self.transform: img = self.transform(img)
        return img, label

# ==========================================
# 3. 实验运行主程序
# ==========================================
def run_sparsity_experiment(
    model_name='resnet18', 
    batch_size=32, 
    use_fake_data=False,  # 默认关闭 Fake Data，使用真实数据
    val_txt_path="Dataset/val.txt", 
    img_dir="Dataset"
):
    print(f"=== 启动激活值稀疏性分析: {model_name} ===")
    
    # 1. 准备模型
    # 这里加载的是标准预训练模型用于分析激活值分布
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif model_name == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    else:
        # 简单的模型工厂兜底
        try:
            model = getattr(models, model_name)(weights='DEFAULT')
        except:
            raise ValueError(f"Unknown model: {model_name}")
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 2. 注册 Profiler Hook
    profiler = BitSparsityProfiler(bit_depth=16)
    hooks = []
    
    print("注册 Hooks...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            profiler.layer_types[name] = module.__class__.__name__
            h = module.register_forward_hook(profiler.hook_fn(name))
            hooks.append(h)

    # 3. 准备数据 (修改为适配 Universal_Eval.py 的逻辑)
    if use_fake_data:
        print("注意: 使用伪造的随机数据 (Fake Data) 进行演示。")
        dataset = datasets.FakeData(size=batch_size * 5, 
                                    image_size=(3, 224, 224), 
                                    num_classes=1000, 
                                    transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=batch_size)
    else:
        print(f"正在从 {img_dir} 加载验证集...")
        
        # 确定输入尺寸
        input_size = 299 if model_name == 'inception_v3' else 224
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        transform_pipeline = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])
        
        # 使用自定义 Dataset
        dataset = FlatImageNetDataset(img_dir, val_txt_path, transform=transform_pipeline)
        
        # 如果只想跑部分数据进行快速验证，可以取消下面这行的注释
        # dataset = torch.utils.data.Subset(dataset, range(1000)) 
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 4. 运行推理
    # 如果是真实数据集，我们通常不需要跑完 50000 张，跑一部分（例如 10-20 个 Batch）就足够统计稀疏性分布了
    # 这里设置 max_batches 来控制
    max_batches = 20 
    print(f"开始推理 (Max Batches: {max_batches})...")
    
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(loader, total=min(len(loader), max_batches))):
            if i >= max_batches: break
            if torch.cuda.is_available():
                images = images.cuda()
            model(images)
    
    # 5. 移除 Hooks
    for h in hooks:
        h.remove()

    # 6. 生成报告
    df = profiler.report()
    
    avg_sparsity = df[[col for col in df.columns if 'Sparsity' in col]].mean()
    
    print("\n" + "="*50)
    print(f"激活值稀疏性分析报告 (Model: {model_name})")
    print("-" * 50)
    print(f"总产生的激活值数量 (INT16 Words): {profiler.total_activations_count:,}")
    print(f"相当于数据量: {profiler.total_activations_count * 2 / 1024 / 1024 / 1024:.2f} GB (16-bit)")
    print("-" * 50)
    print("各粒度下的全模型平均稀疏性 (Zero Slice %):")
    print(avg_sparsity.to_string())
    print("-" * 50)
    
    csv_name = f"{model_name}_activation_sparsity.csv"
    df.to_csv(csv_name, index=False)
    print(f"详细层级数据已保存至: {csv_name}")

if __name__ == "__main__":
    # 配置你的路径
    VAL_TXT = "Dataset/val.txt"
    VAL_IMG_DIR = "Dataset"
    
    # 运行
    run_sparsity_experiment(
        model_name='vgg16', 
        batch_size=64, 
        use_fake_data=False,  # 设为 False 以使用真实数据
        val_txt_path=VAL_TXT,
        img_dir=VAL_IMG_DIR
    )