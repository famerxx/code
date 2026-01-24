import os
import cv2
import torch
import numpy as np
import random
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 1. 路径与参数配置 ---
# 权重文件
checkpoint = r"./checkpoints/sam2.1_hiera_base_plus.pt"
# 模型配置文件
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
# 数据集
voc_root = r"dataset/VOCdevkit/VOC2012/JPEGImages"

RESIZE_HEIGHT = 850  # 设定显示高度，宽度会等比例缩放，解决模糊问题
SAMPLE_COUNT = 20  # 随机抽取的图片数量

# --- 2. 加载模型 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在加载 SAM2 模型至 {device}...")
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

# --- 3. 交互全局变量 ---
drawing_box = False
ix, iy = -1, -1
current_box = None
current_points = []
current_labels = []
last_score = None  # 用于记录最近一次预测的得分


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing_box, current_box, current_points, current_labels
    # 左键：添加点
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        current_labels.append(1)
    # 右键：画框
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing_box = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing_box:
        current_box = [ix, iy, x, y]
    elif event == cv2.EVENT_RBUTTONUP:
        drawing_box = False
        current_box = [ix, iy, x, y]


# --- 4. 运行主逻辑 ---
if not os.path.exists(voc_root):
    print(f"错误：路径不存在 {voc_root}")
    exit()

all_images = [f for f in os.listdir(voc_root) if f.endswith('.jpg')]
selected_images = random.sample(all_images, min(SAMPLE_COUNT, len(all_images)))

window_name = "SAM2_VOC_HighRes_Scored"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(window_name, mouse_callback)

print("\n操作指南：")
print("- [左键点击] 添加点；[右键拖拽] 画矩形框")
print("- [Enter] 执行分割并显示得分")
print("- [Space] 随机下一张图；[Esc] 退出")

for idx, img_name in enumerate(selected_images):
    # 图像预处理：高质量等比例缩放
    raw_img = cv2.imread(os.path.join(voc_root, img_name))
    h, w = raw_img.shape[:2]
    scale = RESIZE_HEIGHT / h
    new_w = int(w * scale)
    scaled_img = cv2.resize(raw_img, (new_w, RESIZE_HEIGHT), interpolation=cv2.INTER_CUBIC)

    display_img = scaled_img.copy()
    current_points, current_labels, current_box = [], [], None
    last_score = None

    predictor.set_image(cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB))

    while True:
        canvas = display_img.copy()

        # 1. 绘制得分 UI
        if last_score is not None:
            score_text = f"Score: {last_score:.2f}"
            # 绘制背景文本框
            cv2.rectangle(canvas, (10, 10), (220, 60), (0, 0, 0), -1)
            cv2.putText(canvas, score_text, (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

        # 2. 绘制交互提示符
        for pt in current_points:
            cv2.drawMarker(canvas, tuple(pt), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
        if current_box:
            cv2.rectangle(canvas, (current_box[0], current_box[1]),
                          (current_box[2], current_box[3]), (0, 0, 255), 2)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # Enter 执行预测
            pts = np.array(current_points) if current_points else None
            lbls = np.array(current_labels) if current_labels else None
            box = np.array(current_box) if current_box else None

            if pts is None and box is None: continue

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(
                    point_coords=pts, point_labels=lbls, box=box, multimask_output=False
                )

            # 更新得分和掩码
            last_score = scores[0]
            mask = masks[0].squeeze()

            # 渲染掩码
            color_mask = np.zeros_like(scaled_img)
            color_mask[mask.astype(bool)] = [255, 144, 30]  # 蓝色掩码
            display_img = cv2.addWeighted(display_img, 0.7, color_mask, 0.3, 0)
            print(f"[{img_name}] 分割完成，得分: {last_score:.4f}")

        elif key == 32:  # Space 下一张
            break
        elif key == 27:  # Esc 退出
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()