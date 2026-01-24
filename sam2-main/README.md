# SAM2复现

---

### 一、环境配置

---

项目实验环境处理器i9-12900，GPU3060laptop，内存16G，系统Win11，cuda版本12.3

下载对应的python,GPU torch,SAM2源文件，官方权重文件

1. 安装python3.13

   ~~~python 
   https://www.python.org/ftp/python/3.13.11/python-3.13.11-amd64.exe
   ~~~

2. 安装torch2.9.1+cu126

   ~~~python 
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ~~~

3. 安装SAM2源文件 https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip

4. 安装官方权重文件 https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt 解压在项目根目录checkpoints下

### 二、数据集准备

---

图片数据集使用COCO2014，下载图片文件注解文件

1. 图片数据下载（包括训练集，测试集，评估数据集）http://images.cocodataset.org/zips/train2014.zip 解压至项目根目录 dataset下

2. 注解文件下载http://images.cocodataset.org/annotations/annotations_trainval2014.zip 解压至项目根目录 dataset下

视频数据集使用 DAVIS2016

1. 视频数据集下载（包含图片数据和注解文件）https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip 解压至项目根目录 dataset下

### 三、必要依赖安装

~~~python 
# 基础核心包
pip install torchvision0.24.1+cu126
pip install timm1.0.24
# sam关键模块
pip install numpy2.2.6
pip install opencv4.12.0.88
pip install Pillow12.0.0
pip install matplotlib3.10.8
~~~

### 四、demo运行

图片分割

在项目目录 [sam2-main](D:\APP\sam2-main) 下运行 [sam2_train_infer_eval_image.py](D:\APP\sam2-main\sam2_train_infer_eval_image.py) 文件

视频动态分割

在项目目录[sam2-main](D:\APP\sam2-main) 下运行 [sam2_train_infer_eval_video.py](D:\APP\sam2-main\sam2_train_infer_eval_video.py) 文件，产生的运行结果在项目目录output/sam_video下

# CLIP-SAM2下游任务

### 一、环境配置

1. 第一部分环境配置部分同上

2. CLIP项目下载https://github.com/openai/CLIP/archive/refs/heads/main.zip

3. 注册为全局文件，进入解压的CLIP文件根目录，执行以下命令为系统安装CLIP环境

   ~~~python
   pip install -e .
   ~~~

### 二、数据集准备

1. 图片数据集使用COCO2014同SAM2项目一致
2. RefCOCO指代文本文件下载，解压至项目根目录 dataset下

> REFCOCO https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
>
> REFCOCO+ https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
>
> REFCOCOg https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip

### 三、必要依赖安装

依赖环境与版本同SAM2一致

### 四、demo运行

在项目目录 [sam2-main](D:\APP\sam2-main) 下运行 [clip-sam2_train_infer_eval_image.py](clip-sam2_train_infer_eval_image.py) 文件，产生的运行结果在项目目录output/clip_sam_image下，消融实验结果在项目目录output/vis_results下