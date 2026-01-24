import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor

# --- 1. 基础路径配置 ---
# 数据集
davis_root = r"dataset/DAVIS"
# 权重文件
checkpoint = r"./checkpoints/sam2.1_hiera_base_plus.pt"
# 模型配置
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
# 视频名称
video_name = "aerobatics"
# 输出路径
output_path = f"output/sam_video/triview_{video_name}_live_with_plt.mp4"

res_mode = "Full-Resolution"
video_in_dir = os.path.join(davis_root, "JPEGImages", res_mode, video_name)
video_gt_dir = os.path.join(davis_root, "Annotations", res_mode, video_name)

# --- 2. 尺寸严格控制 ---
test_img = cv2.imread(os.path.join(video_in_dir, os.listdir(video_in_dir)[0]))
orig_h, orig_w = test_img.shape[:2]
scale_factor = 0.2
w, h = int(orig_w * scale_factor), int(orig_h * scale_factor)
if w % 2 != 0: w -= 1
if h % 2 != 0: h -= 1
total_width = w * 3


def overlay_mask(img, mask, color, alpha=0.5):
    if mask is None or np.sum(mask) == 0: return img.copy()
    out = img.copy()
    out[mask.astype(bool)] = color
    return cv2.addWeighted(img, 1 - alpha, out, alpha, 0)


# --- 3. 准备静态参考帧 ---
gt_available = {}
for f in os.listdir(video_gt_dir):
    if f.endswith('.png'): gt_available[os.path.splitext(f)[0]] = os.path.join(video_gt_dir, f)

first_frame_name = sorted(gt_available.keys())[0]
first_raw = cv2.imread(os.path.join(video_in_dir, first_frame_name + ".jpg"))
first_gt = cv2.imread(gt_available[first_frame_name], cv2.IMREAD_GRAYSCALE)
target_id = int(np.unique(first_gt)[np.unique(first_gt) > 0][0])

ref_frame_res = cv2.resize(first_raw, (w, h))
ref_mask_res = cv2.resize((first_gt == target_id).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
center_static_view = overlay_mask(ref_frame_res, ref_mask_res, [0, 255, 0])
cv2.putText(center_static_view, f"REF ID:{target_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# --- 4. 初始化 SAM2 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
inference_state = predictor.init_state(video_path=video_in_dir)
predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=target_id,
                       mask=(first_gt == target_id).astype(np.uint8))

# --- 5. 视频写入与窗口初始化 ---
input_frame_files = sorted([f for f in os.listdir(video_in_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
input_frame_basenames = [os.path.splitext(f)[0] for f in input_frame_files]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_path, fourcc, 24, (total_width, h))

# 设置实时预览窗口
win_name = "SAM2 Real-time Tri-View"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, total_width, h)

iou_history = []

# --- 6. 执行推理（实时显示 + 数据收集） ---
print(f"🚀 正在处理视频并实时显示...")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        raw_frame = cv2.imread(os.path.join(video_in_dir, input_frame_files[out_frame_idx]))
        frame_resized = cv2.resize(raw_frame, (w, h))

        # [左] SAM2 推理图
        pred_mask_full = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
        pred_mask_res = cv2.resize(pred_mask_full.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        left_view = overlay_mask(frame_resized, pred_mask_res, [0, 165, 255])

        # [右] 原图
        right_view = frame_resized.copy()

        # 计算 IoU 数据
        curr_basename = input_frame_basenames[out_frame_idx]
        iou = 0.0
        if curr_basename in gt_available:
            gt_data = cv2.imread(gt_available[curr_basename], cv2.IMREAD_GRAYSCALE)
            gt_mask_res = cv2.resize((gt_data == target_id).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            inter = np.logical_and(pred_mask_res, gt_mask_res > 0).sum()
            union = np.logical_or(pred_mask_res, gt_mask_res > 0).sum()
            iou = inter / union if union > 0 else 1.0
        iou_history.append(iou)

        # 标注
        cv2.putText(left_view, f"SAM2 IoU:{iou:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(right_view, f"Frame {out_frame_idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 拼接与实时显示
        combined = np.hstack((left_view, center_static_view, right_view))
        out_video.write(combined)

        cv2.imshow(win_name, combined)
        if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 键可以提前退出
            break

out_video.release()
cv2.destroyAllWindows()
print(f"✅ 视频处理完成。")

# --- 7. 处理结束后，使用 Matplotlib 仅显示 IoU 统计图 ---
print("📊 正在弹出 IoU 统计图表...")

plt.figure(figsize=(10, 5))
plt.plot(iou_history, color='tab:orange', linewidth=2, label='Frame IoU')
plt.axhline(y=np.mean(iou_history), color='red', linestyle='--', label=f'mIoU: {np.mean(iou_history):.4f}')

plt.title(f"Segmentation Accuracy Analysis - {video_name}", fontsize=14)
plt.xlabel("Frame Index")
plt.ylabel("IoU Score")
plt.ylim(0, 1.1)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()

# 弹出窗口
plt.show()