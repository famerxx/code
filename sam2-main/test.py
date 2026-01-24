import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
import clip
import random
import pickle
import torchvision
import textwrap

# ================= 🔧 路径配置 =================
# COCO 标注
ANN_PATH = r"D:\Data\refcoco\annotations_trainval2014\annotations\instances_train2014.json"
# 图片根目录
IMG_BASE_DIR = r"D:\Data\refcoco\train2014\train2014"
# RefCOCOg 数据目录
REF_DATA_DIR = r"D:\Data\refcoco\refcocog"
REF_FILE_NAME = "refs(google).p"

# 结果保存路径
SAVE_DIR = "output/vis_results"
os.makedirs(SAVE_DIR, exist_ok=True)
# ===============================================

# --- 1. NMS 安全补丁 ---
try:
    from torchvision.ops import nms as _orig_nms


    def safe_nms(boxes, scores, iou_threshold):
        return _orig_nms(boxes.cpu(), scores.cpu(), iou_threshold).to(boxes.device)


    torchvision.ops.nms = safe_nms
    print("🛡️ NMS 安全补丁已激活")
except:
    pass

# --- 2. 模型初始化 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 初始化设备: {device}")

model_clip, preprocess = clip.load("ViT-B/32", device=device)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = r"./checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
mask_generator = SAM2AutomaticMaskGenerator(sam2_model)


# --- 3. 数据加载 ---
def load_ref_data():
    path = os.path.join(REF_DATA_DIR, REF_FILE_NAME)
    if not os.path.exists(path):
        print(f"❌ 错误: 找不到文件 {path}")
        return None
    with open(path, 'rb') as f:
        refs = pickle.load(f)
    # 过滤验证集
    val_refs = [r for r in refs if r['split'] == 'val']
    print(f"✅ 数据加载成功: 共 {len(val_refs)} 条验证集数据")
    return val_refs


def resolve_target(coco, ref_item):
    # --- 核心修复逻辑 ---
    # 不要用 ref_item['file_name']，因为它带有 _annID 后缀
    # 我们直接用 image_id 重构标准文件名
    img_id = ref_item['image_id']
    fname = f"COCO_train2014_{img_id:012d}.jpg"

    img_path = os.path.join(IMG_BASE_DIR, fname)
    if not os.path.exists(img_path):
        # 尝试去上一级目录找 (防止目录结构差异)
        img_path = os.path.join(os.path.dirname(IMG_BASE_DIR), fname)

    if not os.path.exists(img_path):
        # print(f"找不到图片: {img_path}") # 调试用
        return None, None

    # 获取真值 Mask
    try:
        ann_id = ref_item['ann_id']
        ann = coco.loadAnns(ann_id)[0]
        gt_mask = coco.annToMask(ann)
        return img_path, gt_mask
    except Exception:
        return None, None


# --- 4. 主循环 ---
def run_main(num_samples=20):
    print("⏳ 初始化 COCO API...")
    coco = COCO(ANN_PATH)

    refs = load_ref_data()
    if refs is None: return

    sampled_refs = random.sample(refs, min(num_samples * 2, len(refs)))
    results = []
    processed_count = 0

    print(f"🏁 开始测试 (目标: {num_samples} 张)...")
    pbar = tqdm(total=num_samples)

    for ref in sampled_refs:
        if processed_count >= num_samples: break

        # 提取文本
        if 'sentences' not in ref or not ref['sentences']: continue
        text_query = ref['sentences'][0]['sent']

        # 获取路径和真值
        img_path, gt_mask = resolve_target(coco, ref)
        if img_path is None: continue

        try:
            image = Image.open(img_path).convert("RGB")
            img_np = np.array(image)

            # === SAM2 ===
            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                masks = mask_generator.generate(img_np)
            if not masks: continue

            # === CLIP ===
            text_token = clip.tokenize([text_query[:77]]).to(device)
            with torch.no_grad():
                text_feat = model_clip.encode_text(text_token)
                text_feat /= text_feat.norm(dim=-1, keepdim=True)

                scores = []
                for m in masks:
                    x, y, w, h = [int(v) for v in m['bbox']]
                    pad = 15
                    crop = image.crop((max(0, x - pad), max(0, y - pad), min(image.width, x + w + pad),
                                       min(image.height, y + h + pad)))
                    img_in = preprocess(crop).unsqueeze(0).to(device)
                    img_feat = model_clip.encode_image(img_in)
                    img_feat /= img_feat.norm(dim=-1, keepdim=True)
                    scores.append((img_feat @ text_feat.T).item())

            best_idx = np.argmax(scores)
            pred_mask = masks[best_idx]['segmentation']

            # === 指标 ===
            inter = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = inter / union if union > 0 else 0
            results.append(iou)

            # === 可视化 ===
            plt.figure(figsize=(12, 7))

            # 左图
            plt.subplot(1, 2, 1)
            plt.imshow(img_np)
            color_pred = np.array([1.0, 0.0, 0.0, 0.65])
            h, w = pred_mask.shape[-2:]
            mask_vis_pred = pred_mask.reshape(h, w, 1) * color_pred.reshape(1, 1, -1)
            plt.imshow(mask_vis_pred)

            y_ind, x_ind = np.where(pred_mask > 0)
            if len(y_ind) > 0:
                rect = plt.Rectangle((x_ind.min(), y_ind.min()), x_ind.max() - x_ind.min(), y_ind.max() - y_ind.min(),
                                     linewidth=2, edgecolor='yellow', facecolor='none')
                plt.gca().add_patch(rect)

            wrapped_text = "\n".join(textwrap.wrap(text_query, width=40))
            hit_status = "✅ HIT" if iou > 0.5 else "❌ MISS"

            plt.title(f"Prompt:\n{wrapped_text}\n\nIoU: {iou:.2f} | {hit_status}",
                      fontsize=11, fontweight='bold', color='blue', loc='left')
            plt.axis('off')

            # 右图
            plt.subplot(1, 2, 2)
            plt.imshow(img_np)
            color_gt = np.array([0.0, 1.0, 0.0, 0.5])
            mask_vis_gt = gt_mask.reshape(h, w, 1) * color_gt.reshape(1, 1, -1)
            plt.imshow(mask_vis_gt)
            plt.title("Ground Truth", fontsize=12)
            plt.axis('off')

            save_path = f"{SAVE_DIR}/vis_{processed_count}.jpg"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()

            processed_count += 1
            pbar.update(1)
            pbar.set_description(f"IoU: {iou:.2f}")

        except Exception as e:
            continue

    pbar.close()
    if results:
        print(f"\n📊 最终成绩 (Device: {device})")
        print(f"✅ mIoU: {np.mean(results):.4f}")
        print(f"🎯 成功率: {np.mean(np.array(results) > 0.5):.2%}")
        print(f"📁 结果已保存至: {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    run_main(20)