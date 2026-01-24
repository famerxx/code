import os
import sys

sys.path.append(r"D:\APP\CLIP-main")

import clip
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import random
from tqdm import tqdm
import time  # 新增：用于计算耗时

# ================= ⚙️ 配置区域 =================
TARGET_CATEGORY = 'cat'
TEXT_PROMPT = f"the {TARGET_CATEGORY}"
NUM_SAMPLES = 20
# 描述文件
ANN_PATH = r"dataset/refcoco/annotations_trainval2014/annotations/instances_train2014.json"
# 数据集
IMG_BASE_DIR = r"dataset/refcoco/train2014/train2014"
# 输出目录
SAVE_DIR = "output/clip_sam_image"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE_ID = 0  # 指定GPU编号
# ===============================================

# --- 解决matplotlib字体警告（新增）---
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial']  # 增加字体 fallback
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'

# --- 初始化设备（GPU模式）---
if torch.cuda.is_available():
    device = f"cuda:{DEVICE_ID}"
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(DEVICE_ID)
    print(f"✅ 检测到CUDA，使用GPU {DEVICE_ID} 运行 (torch版本: {torch.__version__}, CUDA: {torch.cuda.is_available()})")
else:
    device = "cpu"
    print("⚠️ 未检测到CUDA，自动降级为CPU模式")

print(f"⚙️ 初始化: 寻找 '{TEXT_PROMPT}' ({device}模式)...")

# --- 加载模型 ---
model_clip, preprocess = clip.load("ViT-B/32", device=device)
model_clip.eval()

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = r"./checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
mask_generator_predictor = SAM2ImagePredictor(sam2_model)


# --- 工具函数 ---
def calculate_box_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    b1_x1, b1_y1, b1_x2, b1_y2 = x1, y1, x1 + w1, y1 + h1
    b2_x1, b2_y1, b2_x2, b2_y2 = x2, y2, x2 + w2, y2 + h2
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2: return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / (b1_area + b2_area - inter_area)


def calculate_mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0


def get_clip_heatmap_center(image, text):
    if image.size[0] == 0 or image.size[1] == 0:
        print("⚠️ 图片尺寸异常，返回默认中心点")
        return np.array([[image.size[0] // 2, image.size[1] // 2]])

    h, w = image.size[1], image.size[0]
    small_img = image.resize((224, 224))
    grid_h, grid_w = 7, 7
    patch_h, patch_w = 224 // grid_h, 224 // grid_w
    patches = []

    for i in range(grid_h):
        for j in range(grid_w):
            patch = small_img.crop((j * patch_w, i * patch_h, (j + 1) * patch_w, (i + 1) * patch_h))
            patches.append(preprocess(patch))

    image_input = torch.stack(patches).to(device, dtype=torch.float32)
    text_token = clip.tokenize([text]).to(device)

    with torch.no_grad():
        img_feats = model_clip.encode_image(image_input)
        txt_feats = model_clip.encode_text(text_token)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
        sim = (img_feats @ txt_feats.T).squeeze()

    sim_np = sim.cpu().numpy()
    sim_np = np.nan_to_num(sim_np, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = sim_np.reshape(grid_h, grid_w).astype(np.float32)

    if heatmap.size == 0 or w == 0 or h == 0:
        print("⚠️ 热力图尺寸异常，返回默认中心点")
        return np.array([[w // 2, h // 2]])

    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_blur = cv2.GaussianBlur(heatmap, (15, 15), 0)
    (_, _, _, maxLoc) = cv2.minMaxLoc(heatmap_blur)

    if maxLoc[0] < 0 or maxLoc[1] < 0 or maxLoc[0] >= w or maxLoc[1] >= h:
        maxLoc = (w // 2, h // 2)

    return np.array([maxLoc])


# --- 主循环（重点优化tqdm）---
def run_evaluation():
    coco = COCO(ANN_PATH)
    cat_ids = coco.getCatIds(catNms=[TARGET_CATEGORY])
    if not cat_ids:
        print(f"❌ COCO 中没有 '{TARGET_CATEGORY}' 这个类别")
        return
    img_ids = coco.getImgIds(catIds=cat_ids)
    sample_ids = random.sample(img_ids, min(NUM_SAMPLES, len(img_ids)))
    total_samples = len(sample_ids)
    print(f"🚀 开始评估 {total_samples} 张图片 (目标类别: {TARGET_CATEGORY})...")

    mask_ious = []
    box_ious = []
    success_mask_count = 0
    success_box_count = 0
    failed_count = 0  # 新增：统计失败数
    start_time = time.time()  # 新增：记录开始时间

    # --- 优化tqdm进度条配置 ---
    pbar = tqdm(
        sample_ids,
        total=total_samples,
        desc=f"📊 评估进度 [{TARGET_CATEGORY}]",  # 自定义描述
        bar_format="{l_bar}{bar:20}{r_bar}",  # 进度条宽度固定20字符
        colour="green",  # 进度条颜色（green/red/blue/yellow）
        ncols=100,  # 进度条总宽度
        unit="img",  # 单位名称
        dynamic_ncols=False,  # 固定宽度，避免闪烁
        leave=True  # 完成后保留进度条
    )

    for idx, img_id in enumerate(pbar):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMG_BASE_DIR, img_info['file_name'])

        if not os.path.exists(img_path):
            failed_count += 1
            # 更新进度条描述（实时显示关键指标）
            pbar.set_postfix({
                "mIoU": f"{np.mean(mask_ious):.4f}" if mask_ious else "0.0000",
                "成功率": f"{success_mask_count / (idx + 1):.2%}" if (idx + 1) > 0 else "0.00%",
                "失败数": failed_count,
                "耗时": f"{time.time() - start_time:.1f}s"
            })
            pbar.update(1)
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            if image.size[0] < 10 or image.size[1] < 10:
                failed_count += 1
                pbar.set_postfix({
                    "mIoU": f"{np.mean(mask_ious):.4f}" if mask_ious else "0.0000",
                    "成功率": f"{success_mask_count / (idx + 1):.2%}" if (idx + 1) > 0 else "0.00%",
                    "失败数": failed_count,
                    "耗时": f"{time.time() - start_time:.1f}s"
                })
                pbar.update(1)
                continue

            img_np = np.array(image)
            input_point = get_clip_heatmap_center(image, TEXT_PROMPT)
            mask_generator_predictor.set_image(img_np)
            masks, scores, _ = mask_generator_predictor.predict(
                point_coords=input_point,
                point_labels=np.array([1]),
                multimask_output=False
            )
            pred_mask = masks[0]

            y_ind, x_ind = np.where(pred_mask > 0)
            if len(y_ind) > 0:
                pred_bbox = [x_ind.min(), y_ind.min(), x_ind.max() - x_ind.min(), y_ind.max() - y_ind.min()]
            else:
                pred_bbox = [0, 0, 0, 0]

            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
            anns = coco.loadAnns(ann_ids)
            best_m_iou = 0.0
            best_b_iou = 0.0

            for ann in anns:
                gt_mask = coco.annToMask(ann)
                m = calculate_mask_iou(pred_mask, gt_mask)
                b = calculate_box_iou(pred_bbox, ann['bbox'])
                if m > best_m_iou: best_m_iou = m
                if b > best_b_iou: best_b_iou = b

            if best_m_iou >= 0 and best_b_iou >= 0:
                mask_ious.append(best_m_iou)
                box_ious.append(best_b_iou)
                if best_m_iou > 0.5: success_mask_count += 1
                if best_b_iou > 0.5: success_box_count += 1

            # 可视化（替换emoji为文字，解决字体警告）
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(img_np)
            color = np.array([1.0, 0.0, 0.0, 0.6])
            h_mask, w_mask = pred_mask.shape[-2:]
            mask_vis = pred_mask.reshape(h_mask, w_mask, 1) * color.reshape(1, 1, -1)
            plt.imshow(mask_vis)
            plt.scatter(input_point[0][0], input_point[0][1], c='yellow', marker='*', s=200, edgecolors='black')
            # 核心修改：替换emoji为文字（Hit/Miss）
            is_hit = "Hit" if best_m_iou > 0.5 else "Miss"
            title_text = f"Prompt: \"{TEXT_PROMPT}\"\nMask IoU: {best_m_iou:.2f} | {is_hit}"
            plt.title(title_text, fontsize=12, fontweight='bold', color='blue')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            ov2 = img_np.copy()
            for ann in anns:
                gm = coco.annToMask(ann)
                ov2[gm > 0] = [0, 255, 0]
            plt.imshow(cv2.addWeighted(img_np, 0.6, ov2, 0.4, 0))
            plt.title(f"Ground Truth ({TARGET_CATEGORY})", fontsize=12)
            plt.axis('off')

            plt.savefig(f"{SAVE_DIR}/eval_{idx}_{TARGET_CATEGORY}.jpg", bbox_inches='tight')
            plt.close()

            torch.cuda.empty_cache()

            # --- 实时更新进度条后缀（核心优化）---
            current_progress = idx + 1
            avg_miou = np.mean(mask_ious) if mask_ious else 0.0
            success_rate = success_mask_count / current_progress if current_progress > 0 else 0.0
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / current_progress) * (
                        total_samples - current_progress) if current_progress > 0 else 0  # 预计剩余时间

            pbar.set_postfix({
                "mIoU": f"{avg_miou:.4f}",
                "成功率": f"{success_rate:.2%}",
                "失败数": failed_count,
                "耗时": f"{elapsed_time:.1f}s",
                "剩余": f"{eta:.1f}s"  # 新增：预计剩余时间
            })

        except Exception as e:
            failed_count += 1
            print(f"\n⚠️ 处理图片 {img_id} 时出错: {str(e)}")
            torch.cuda.empty_cache()
            # 出错时也更新进度条
            current_progress = idx + 1
            avg_miou = np.mean(mask_ious) if mask_ious else 0.0
            success_rate = success_mask_count / current_progress if current_progress > 0 else 0.0
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / current_progress) * (total_samples - current_progress) if current_progress > 0 else 0

            pbar.set_postfix({
                "mIoU": f"{avg_miou:.4f}",
                "成功率": f"{success_rate:.2%}",
                "失败数": failed_count,
                "耗时": f"{elapsed_time:.1f}s",
                "剩余": f"{eta:.1f}s"
            })

    pbar.close()

    # 计算最终指标
    total_time = time.time() - start_time
    m_iou = np.mean(mask_ious) if mask_ious else 0.0
    b_iou = np.mean(box_ious) if box_ious else 0.0
    success_rate = success_mask_count / total_samples if total_samples > 0 else 0.0
    avg_time_per_img = total_time / total_samples if total_samples > 0 else 0.0

    # 打印美化后的结果
    print("\n" + "=" * 60)
    print(f"📊 最终评估结果 (目标: '{TEXT_PROMPT}', 设备: {device})")
    print("=" * 60)
    print(f"📈 分割精度 (mIoU)       : {m_iou:.4f}")
    print(f"📏 框精度 (IoU@0.5)      : {b_iou:.4f}")
    print(f"✅ 成功率 (Mask IoU>0.5) : {success_rate:.2%} ({success_mask_count}/{total_samples})")
    print(f"❌ 失败数                : {failed_count}")
    print(f"⏱️  总耗时               : {total_time:.2f}秒")
    print(f"⚡ 单图平均耗时           : {avg_time_per_img:.2f}秒")
    print("=" * 60)
    print(f"📁 结果已保存至: {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    run_evaluation()