import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
import xml.etree.ElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np

# ================= ⚙️ 路径与参数配置 =================
local_repo_path = r"D:\APP\computerView\dinov2-main"
# 权重文件
local_weights_path = os.path.join(local_repo_path, "dinov2_vits14_pretrain.pth")
# 数据集
voc_root = r"D:\Data\VOCdevkit\VOC2012"
# 批次大小
BATCH_SIZE = 32
# 224*224 尺寸
IMG_SIZE = 224
EPOCHS = 1
LEARNING_RATE = 0.001
NUM_WORKERS = 4

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]
class_to_idx = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 📂 数据集定义 =================
class VOCRealDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root = root_dir
        self.transform = transform
        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.ann_dir = os.path.join(root_dir, "Annotations")
        txt_name = "train.txt" if is_train else "val.txt"
        txt_path = os.path.join(root_dir, "ImageSets", "Main", txt_name)

        self.ids = []
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                self.ids = [line.strip() for line in f.readlines()]
        else:
            all_files = [f[:-4] for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
            split = int(len(all_files) * 0.8)
            self.ids = all_files[:split] if is_train else all_files[split:]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")
        target_class = 0
        if os.path.exists(ann_path):
            try:
                tree = ET.parse(ann_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in class_to_idx:
                        target_class = class_to_idx[name]
                        break
            except:
                pass

        if self.transform: image = self.transform(image)
        return image, target_class


# ================= 🧠 模型定义 =================
class DINOv2LocalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.printed_shapes = False

    def load_weights(self):
        print(f"\n📦 正在加载本地 DINOv2...", flush=True)
        if not os.path.exists(os.path.join(local_repo_path, 'hubconf.py')):
            print(f"❌ 错误：在 {local_repo_path} 下找不到 hubconf.py")
            sys.exit(1)
        if not os.path.exists(local_weights_path):
            print(f"❌ 错误：找不到权重文件 {local_weights_path}")
            sys.exit(1)

        self.backbone = torch.hub.load(local_repo_path, 'dinov2_vits14', source='local', pretrained=False)
        state_dict = torch.load(local_weights_path, map_location='cpu')
        self.backbone.load_state_dict(state_dict)
        print("   ✅ 模型加载成功！(离线模式)", flush=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.feat_dim = 384
        self.classifier = nn.Linear(self.feat_dim, len(VOC_CLASSES)).to(device)
        self.flatten = nn.Flatten()

    def forward(self, x):
        with torch.no_grad():
            output = self.backbone.forward_features(x)
            global_feat = output["x_norm_clstoken"]
            dense_feat = output["x_norm_patchtokens"]

            if not self.printed_shapes:
                print("\n" + "=" * 50, flush=True)
                print("   🔍 [DINOv2 特征提取验证]", flush=True)
                print(f"   输入图像 Batch : {x.shape} (B, C, H, W)", flush=True)
                print(f"   Global Feature : {global_feat.shape} (B, 384) -> 用于分类", flush=True)
                print(f"   Dense Feature  : {dense_feat.shape} (B, 256, 384) -> 局部细节", flush=True)
                print("=" * 50 + "\n", flush=True)
                self.printed_shapes = True

        x = self.flatten(global_feat)
        return self.classifier(x)


# ================= 📊 绘图函数 1: 训练曲线 =================
def plot_training_curve(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.title('Training Loss');
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.grid(True);
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'r-o', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'g-s', label='Val Acc')
    plt.title('Training & Validation Accuracy');
    plt.xlabel('Epochs');
    plt.ylabel('Accuracy (%)');
    plt.grid(True);
    plt.legend()
    plt.tight_layout()
    save_path = "output/training_curve.png"
    plt.savefig(save_path)
    print(f"\n📈 训练曲线已保存为: {save_path}", flush=True)
    # plt.show() # 如果不想弹窗可以注释掉


# ================= 🎨 绘图函数 2: 随机预测可视化 (新功能) =================
def visualize_predictions(model, dataset, device, num_samples=20):
    print(f"\n🎨 正在抽取 {num_samples} 张图片进行可视化测试...", flush=True)
    model.eval()

    # 随机抽取索引
    indices = torch.randperm(len(dataset))[:num_samples].tolist()

    # 设置画布 (4行5列)
    fig, axes = plt.subplots(4, 5, figsize=(16, 12))
    fig.suptitle(f'Random {num_samples} Predictions (Green=Correct, Red=Wrong)', fontsize=16)

    # 图像反归一化参数 (ImageNet Standard)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.224])

    for idx, ax in zip(indices, axes.flat):
        image, label = dataset[idx]

        # 预测
        input_tensor = image.unsqueeze(0).to(device)  # (1, 3, 224, 224)
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()

        # 处理图片用于显示 (Tensor -> Numpy -> 反归一化)
        img_display = image.permute(1, 2, 0).cpu().numpy()
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)  # 限制在 0-1 之间

        # 显示图片
        ax.imshow(img_display)

        # 设置标题颜色
        color = 'green' if pred == label else 'red'
        title_text = f"P: {VOC_CLASSES[pred]}\nT: {VOC_CLASSES[label]}"
        ax.set_title(title_text, color=color, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    save_path = "output/prediction_gallery.png"
    plt.savefig(save_path)
    print(f"📸 预测可视化已保存为: {save_path}", flush=True)
    plt.show()


# ================= 🚀 主程序 =================
def main():
    print(f"🚀 运行设备: {device} | 线程数: {NUM_WORKERS}", flush=True)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224]),
    ])

    print("\n📂 加载数据...", flush=True)
    train_set = VOCRealDataset(voc_root, transform=transform, is_train=True)
    val_set = VOCRealDataset(voc_root, transform=transform_val, is_train=False)

    print(f"   --------------------------------", flush=True)
    print(f"   🖼️  训练集: {len(train_set)} 张", flush=True)
    print(f"   🖼️  验证集: {len(val_set)} 张", flush=True)
    print(f"   🏷️  类别数: {len(VOC_CLASSES)} 类", flush=True)
    print(f"   --------------------------------", flush=True)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    model = DINOv2LocalModel(len(VOC_CLASSES))
    model.load_weights()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=LEARNING_RATE, momentum=0.9)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    print("⚡ 正在执行特征提取检查 (Dummy Pass)...", flush=True)
    with torch.no_grad():
        dummy_input = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(device)
        model(dummy_input)

    time.sleep(1.0)

    print(f"⚡ 开始训练 (共 {EPOCHS} 轮, {NUM_WORKERS} 线程)...", flush=True)
    time.sleep(0.5)

    for epoch in range(EPOCHS):
        print(f"\n[ Epoch {epoch + 1}/{EPOCHS} ]", flush=True)
        time.sleep(0.2)

        # --- Train ---
        model.train()
        running_loss = 0.0;
        train_correct = 0;
        train_total = 0

        with tqdm(train_loader, desc="Train", unit="batch", ncols=100, leave=True, file=sys.stdout) as train_bar:
            for images, labels in train_bar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                train_bar.set_postfix(
                    {"acc": f"{100. * train_correct / train_total:.1f}%", "loss": f"{loss.item():.2f}"})

        epoch_loss = running_loss / train_total
        epoch_acc = 100. * train_correct / train_total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # --- Val ---
        model.eval()
        val_correct = 0;
        val_total = 0

        with tqdm(val_loader, desc="Val  ", unit="batch", ncols=100, leave=True, file=sys.stdout) as val_bar:
            with torch.no_grad():
                for images, labels in val_bar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    val_bar.set_postfix({"acc": f"{100. * val_correct / val_total:.1f}%"})

        val_acc = 100. * val_correct / val_total
        history['val_acc'].append(val_acc)

        print("", flush=True)
        time.sleep(0.05)
        print(f"📊 结果: Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%", flush=True)
        sys.stdout.flush()
        time.sleep(0.5)

    print("\n✅ 训练完成！正在生成图表...", flush=True)
    plot_training_curve(history)

    # ✅✅✅ 这里的调用是新增的 ✅✅✅
    # 在最后调用随机可视化函数
    visualize_predictions(model, val_set, device, num_samples=20)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()