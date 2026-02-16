import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm

# ==========================================
# 1. 自定义 Dataset (处理你的 val.txt 结构)
# ==========================================
class FlatImageNetDataset(Dataset):
    def __init__(self, img_dir, val_txt_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        with open(val_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # 注意：ResNet18 默认 label 0-999。
                    # 如果你的 txt 是 0-999，直接用。如果是 1-1000，需要 -1。
                    # 这里假设是标准 ILSVRC2012 label (通常需要校验)
                    self.samples.append((parts[0], int(parts[1])))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        path = os.path.join(self.img_dir, filename)
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

# ==========================================
# 2. 验证逻辑
# ==========================================
def validate_model(weights_path):
    # --- 配置路径 (请修改这里) ---
    VAL_TXT = r"Dataset//val.txt"
    VAL_IMG_DIR = r"Dataset"
    BATCH_SIZE = 128
    WORKERS = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备数据
    print(f"Loading ImageNet from {VAL_IMG_DIR} ...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = FlatImageNetDataset(
        VAL_IMG_DIR, VAL_TXT,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

    # 2. 加载模型结构
    print("Creating ResNet18 model...")
    model = models.resnet18() # 不需要 pretrained=True，因为我们要加载自己的权重
    
    # 3. 加载压缩后的权重
    print(f"Loading compressed weights from {weights_path} ...")
    try:
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model = model.to(DEVICE)
    model.eval()

    # 4. 推理循环
    print("Starting validation...")
    correct_1 = 0
    correct_5 = 0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            
            # Top-k accuracy
            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            correct_1 += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
            correct_5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
            total += targets.size(0)

    print("\n" + "="*40)
    print(f"Validation Results for: {weights_path}")
    print(f"Top-1 Accuracy: {correct_1 / total * 100:.2f}%")
    print(f"Top-5 Accuracy: {correct_5 / total * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    # 确保这个文件名与第一步生成的文件名一致
    validate_model("resnet18_bit_density_compressed.pth")