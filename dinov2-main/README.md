# DINOV2复现

### 一、环境配置

---

项目实验环境处理器i9-12900，GPU3060laptop，内存16G，系统Win11，cuda版本12.3

下载对应的python,GPU torch，官方权重文件

1. 安装python3.13

   ~~~python 
   https://www.python.org/ftp/python/3.13.11/python-3.13.11-amd64.exe
   ~~~

2. 安装torch2.9.1+cu126

   ~~~python 
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ~~~

4. 安装官方权重文件https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth 解压在项目根目录下

### 二、数据集准备

---

图片数据集使用VOC2012，下载图片文件注解文件

1. 图片数据下载（包括训练集，测试集，评估数据集）https://datasets.cms.waikato.ac.nz/ufdl/data/pascalvoc2012/VOCtrainval_11-May-2012.tar解压至项目根目录 dataset下

### 三、必要依赖安装

~~~python 
pip install torch2.9.1
pip install torchvision0.24.1
pip install omegaconf2.3.0
pip install iopath0.1.10
~~~

### 四、demo运行

图片分类

在项目目录 [dinov2-main]() 下运行 [project_voc_local.py](project_voc_local.py)  文件，模型训练曲线与结果可视化图片生成在项目根目录output下
