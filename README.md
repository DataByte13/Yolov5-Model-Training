```python
## This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

!python train.py --resume runs/train/exp1/weights/best.pt --epochs 50


```python
%pip install torch torchvision torchaudio
%pip install pyyaml matplotlib tqdm
!git clone https://github.com/ultralytics/yolov5.git # clone
%cd yolov5
%pip install -qr requirements.txt  # install

import torch
import utils
display = utils.notebook_init()  # checks
```

    YOLOv5 🚀 v7.0-321-g3742ab49 Python-3.10.13 torch-2.1.2 CUDA:0 (Tesla P100-PCIE-16GB, 16276MiB)


    Setup complete ✅ (4 CPUs, 31.4 GB RAM, 5689.4/8062.4 GB disk)



```python
!ls
```


```python
visdrone_yaml = """
path : /kaggle/input/visdrone2019
train : /kaggle/input/visdrone2019/images/train
val : /kaggle/input/visdrone2019/images/val 
names:

  0: pedes #pedestrian

  1: people

  2: bicycle

  3: car

  4: van

  5: truck

  6: tricycle

  7: awntric #awning-tricycle

  8: bus

  9: motor
download: |
  from utils.general import download, Path


  # Download labels
  segments = False  # segment or box labels
  dir = Path(yaml['path'])  # dataset root dir
  url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
  urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels
  download(urls, dir=dir.parent)

  # Download data
  urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
          'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
          'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
  download(urls, dir=dir / 'images', threads=3)

"""

# Save the changes to a new YAML file
with open('/kaggle/working/yolov5/data/cocoCastom.yaml', 'w') as file:
    file.write(visdrone_yaml)

```


```python
!ls
!python train.py --img 640 --batch 16 --epochs 50 --data cocoCastom.yaml --weights yolov5m.pt --cache --name exp1_50ep
#!cp runs/train/exp1_50ep/weights/best.pt runs/train/exp1_50ep/weights/exp1_ep50_best.pt.backup 
```

    CITATION.cff	 README.zh-CN.md  detect.py   pyproject.toml	tutorial.ipynb
    CONTRIBUTING.md  benchmarks.py	  export.py   requirements.txt	utils
    LICENSE		 classify	  hubconf.py  segment		val.py
    README.md	 data		  models      train.py
    2024-06-09 23:49:37.071244: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-06-09 23:49:37.071388: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-06-09 23:49:37.198390: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    [34m[1mtrain: [0mweights=yolov5m.pt, cfg=, data=cocoCastom.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=50, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, evolve_population=data/hyps, resume_evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp1_50ep, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest, ndjson_console=False, ndjson_file=False
    [34m[1mgithub: [0mup to date with https://github.com/ultralytics/yolov5 ✅
    YOLOv5 🚀 v7.0-321-g3742ab49 Python-3.10.13 torch-2.1.2 CUDA:0 (Tesla P100-PCIE-16GB, 16276MiB)
    
    [34m[1mhyperparameters: [0mlr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
    [34m[1mComet: [0mrun 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
    [34m[1mTensorBoard: [0mStart with 'tensorboard --logdir runs/train', view at http://localhost:6006/
    Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt to yolov5m.pt...
    100%|███████████████████████████████████████| 40.8M/40.8M [00:00<00:00, 160MB/s]
    
    Overriding model.yaml nc=80 with nc=10
    
                     from  n    params  module                                  arguments                     
      0                -1  1      5280  models.common.Conv                      [3, 48, 6, 2, 2]              
      1                -1  1     41664  models.common.Conv                      [48, 96, 3, 2]                
      2                -1  2     65280  models.common.C3                        [96, 96, 2]                   
      3                -1  1    166272  models.common.Conv                      [96, 192, 3, 2]               
      4                -1  4    444672  models.common.C3                        [192, 192, 4]                 
      5                -1  1    664320  models.common.Conv                      [192, 384, 3, 2]              
      6                -1  6   2512896  models.common.C3                        [384, 384, 6]                 
      7                -1  1   2655744  models.common.Conv                      [384, 768, 3, 2]              
      8                -1  2   4134912  models.common.C3                        [768, 768, 2]                 
      9                -1  1   1476864  models.common.SPPF                      [768, 768, 5]                 
     10                -1  1    295680  models.common.Conv                      [768, 384, 1, 1]              
     11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     12           [-1, 6]  1         0  models.common.Concat                    [1]                           
     13                -1  2   1182720  models.common.C3                        [768, 384, 2, False]          
     14                -1  1     74112  models.common.Conv                      [384, 192, 1, 1]              
     15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     16           [-1, 4]  1         0  models.common.Concat                    [1]                           
     17                -1  2    296448  models.common.C3                        [384, 192, 2, False]          
     18                -1  1    332160  models.common.Conv                      [192, 192, 3, 2]              
     19          [-1, 14]  1         0  models.common.Concat                    [1]                           
     20                -1  2   1035264  models.common.C3                        [384, 384, 2, False]          
     21                -1  1   1327872  models.common.Conv                      [384, 384, 3, 2]              
     22          [-1, 10]  1         0  models.common.Concat                    [1]                           
     23                -1  2   4134912  models.common.C3                        [768, 768, 2, False]          
     24      [17, 20, 23]  1     60615  models.yolo.Detect                      [10, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [192, 384, 768]]
    Model summary: 291 layers, 20907687 parameters, 20907687 gradients, 48.3 GFLOPs
    
    Transferred 475/481 items from yolov5m.pt
    [34m[1mAMP: [0mchecks passed ✅
    [34m[1moptimizer:[0m SGD(lr=0.01) with parameter groups 79 weight(decay=0.0), 82 weight(decay=0.0005), 82 bias
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
    /opt/conda/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()
    [34m[1mtrain: [0mScanning /kaggle/input/visdrone2019/labels/train... 6471 images, 0 backgr[0m
    [34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/visdrone2019/images/train/0000137_02220_d_0000163.jpg: 1 duplicate labels removed
    [34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/visdrone2019/images/train/0000140_00118_d_0000002.jpg: 1 duplicate labels removed
    [34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/visdrone2019/images/train/9999945_00000_d_0000114.jpg: 1 duplicate labels removed
    [34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/visdrone2019/images/train/9999987_00000_d_0000049.jpg: 1 duplicate labels removed
    [34m[1mtrain: [0mWARNING ⚠️ Cache directory /kaggle/input/visdrone2019/labels is not writeable: [Errno 30] Read-only file system: '/kaggle/input/visdrone2019/labels/train.cache.npy'
    [34m[1mtrain: [0mCaching images (4.9GB ram): 100%|██████████| 6471/6471 [00:29<00:00, 220.[0m
    [34m[1mval: [0mScanning /kaggle/input/visdrone2019/labels/val... 548 images, 0 backgrounds[0m
    [34m[1mval: [0mWARNING ⚠️ Cache directory /kaggle/input/visdrone2019/labels is not writeable: [Errno 30] Read-only file system: '/kaggle/input/visdrone2019/labels/val.cache.npy'
    [34m[1mval: [0mCaching images (0.4GB ram): 100%|██████████| 548/548 [00:03<00:00, 163.37it[0m
    
    [34m[1mAutoAnchor: [0m2.95 anchors/target, 0.933 Best Possible Recall (BPR). Anchors are a poor fit to dataset ⚠️, attempting to improve...
    [34m[1mAutoAnchor: [0mWARNING ⚠️ Extremely small objects found: 29644 of 343201 labels are <3 pixels in size
    [34m[1mAutoAnchor: [0mRunning kmeans for 9 anchors on 342304 points...
    [34m[1mAutoAnchor: [0mEvolving anchors with Genetic Algorithm: fitness = 0.7501: 100%|████[0m
    [34m[1mAutoAnchor: [0mthr=0.25: 0.9995 best possible recall, 5.74 anchors past thr
    [34m[1mAutoAnchor: [0mn=9, img_size=640, metric_all=0.364/0.749-mean/best, past_thr=0.486-mean: 3,5, 4,9, 8,7, 8,14, 16,9, 14,21, 29,16, 34,34, 63,60
    [34m[1mAutoAnchor: [0mDone ✅ (optional: update model *.yaml to use these anchors in the future)
    Plotting labels to runs/train/exp1_50ep/labels.jpg... 
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    Image sizes 640 train, 640 val
    Using 4 dataloader workers
    Logging results to [1mruns/train/exp1_50ep[0m
    Starting training for 50 epochs...
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           0/49      7.11G     0.1244     0.1405    0.04769        293        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.33      0.191      0.114     0.0505
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           1/49      7.11G     0.1069     0.1675    0.03595        641        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.376      0.217       0.16     0.0726
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           2/49      7.12G      0.105     0.1681    0.03304        483        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.428      0.212      0.184     0.0866
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           3/49      7.12G     0.1027     0.1681     0.0316        820        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.542      0.225      0.201     0.0947
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           4/49      7.12G     0.1008     0.1663    0.03034        675        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.49      0.251      0.233      0.114
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           5/49      7.12G    0.09999     0.1663    0.02959        402        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.445      0.262      0.247      0.125
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           6/49      7.12G     0.0991     0.1635    0.02871        481        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.443       0.29      0.264      0.137
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           7/49      7.12G    0.09848     0.1643    0.02837        579        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.371      0.307      0.278      0.147
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           8/49      7.12G    0.09743     0.1627    0.02783        775        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.387      0.308      0.286      0.152
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           9/49      7.12G    0.09717     0.1623    0.02736        449        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.398      0.325      0.302      0.161
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          10/49      7.12G    0.09644     0.1608    0.02688        326        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.401      0.314      0.294      0.158
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          11/49      7.12G    0.09623     0.1597    0.02658        260        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.426      0.323       0.31      0.168
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          12/49      7.12G    0.09583     0.1599    0.02613        643        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.407      0.337      0.317      0.172
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          13/49      7.12G    0.09555     0.1604    0.02602        612        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.42      0.331      0.319      0.173
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          14/49      7.12G    0.09523     0.1577    0.02564        670        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.42      0.338      0.321      0.173
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          15/49      7.12G    0.09505      0.159    0.02543        375        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.433      0.337      0.329      0.181
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          16/49      7.12G    0.09479     0.1573    0.02516        555        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.433      0.337      0.329       0.18
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          17/49      7.12G    0.09434     0.1559     0.0251        575        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.434      0.338      0.334      0.183
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          18/49      7.12G    0.09461     0.1569    0.02486        710        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.447      0.338      0.335      0.183
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          19/49      7.12G    0.09399     0.1586    0.02466        770        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.448      0.349      0.339      0.187
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          20/49      7.12G    0.09391     0.1556    0.02435        572        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.451      0.356      0.346      0.191
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          21/49      7.12G    0.09373     0.1554    0.02439        621        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.456      0.354      0.346      0.192
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          22/49      7.12G    0.09347     0.1545    0.02417        717        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.45      0.351      0.346      0.195
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          23/49      7.12G    0.09307     0.1533    0.02418        661        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.449      0.355      0.347      0.194
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          24/49      7.12G    0.09307      0.154    0.02389        316        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.464      0.341      0.345      0.192
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          25/49      7.12G    0.09306      0.154    0.02388        715        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.455      0.358      0.352      0.196
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          26/49      7.12G    0.09271      0.154     0.0235        684        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.454      0.363      0.356      0.199
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          27/49      7.12G    0.09263     0.1514    0.02334        622        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.451      0.359      0.352      0.198
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          28/49      7.12G    0.09219     0.1527    0.02332        436        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.456      0.367      0.359      0.201
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          29/49      7.12G    0.09214     0.1522    0.02332        624        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.47      0.354      0.358      0.201
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          30/49      7.12G    0.09223     0.1515    0.02309        425        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.46      0.372      0.366      0.205
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          31/49      7.12G    0.09191     0.1503    0.02298        471        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.456      0.366      0.358        0.2
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          32/49      7.12G    0.09187     0.1513    0.02274        752        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.467      0.364      0.363      0.204
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          33/49      7.12G    0.09181     0.1493     0.0229        379        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.484      0.358      0.364      0.205
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          34/49      7.12G    0.09177     0.1507    0.02275        617        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.467      0.372      0.363      0.205
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          35/49      7.12G    0.09123     0.1493    0.02257        536        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.472      0.366      0.364      0.206
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          36/49      7.12G    0.09186     0.1484    0.02248        401        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.474      0.363      0.364      0.206
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          37/49      7.12G    0.09103     0.1502    0.02235        691        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.462      0.378      0.366      0.207
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          38/49      7.12G    0.09116     0.1479    0.02228        574        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.466      0.371      0.365      0.207
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          39/49      7.12G    0.09081     0.1469    0.02222        750        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.477      0.369      0.368      0.209
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          40/49      7.12G    0.09136     0.1465    0.02227        841        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.473      0.375      0.373       0.21
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          41/49      7.12G     0.0905     0.1456    0.02197        576        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.478      0.373       0.37       0.21
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          42/49      7.12G    0.09095     0.1485    0.02188        589        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.482      0.375      0.374      0.214
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          43/49      7.12G    0.09071     0.1431    0.02169        315        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.472       0.38      0.372      0.212
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          44/49      7.12G    0.09037     0.1453    0.02166        792        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.487      0.374      0.375      0.213
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          45/49      7.12G    0.09013     0.1449    0.02159        651        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.486      0.373      0.375      0.214
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          46/49      7.12G    0.09064     0.1444    0.02161        336        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.483      0.375      0.375      0.213
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          47/49      7.12G    0.09007     0.1456    0.02148        384        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.478      0.376      0.373      0.213
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          48/49      7.12G     0.0899     0.1443     0.0214        500        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.478      0.377      0.375      0.215
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          49/49      7.12G    0.08956      0.141    0.02127        628        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.475      0.379      0.375      0.215
    
    50 epochs completed in 2.816 hours.
    Optimizer stripped from runs/train/exp1_50ep/weights/last.pt, 42.2MB
    Optimizer stripped from runs/train/exp1_50ep/weights/best.pt, 42.2MB
    
    Validating runs/train/exp1_50ep/weights/best.pt...
    Fusing layers... 
    Model summary: 212 layers, 20889303 parameters, 0 gradients, 48.0 GFLOPs
                     Class     Images  Instances          P          R      mAP50   WARNING ⚠️ NMS time limit 2.100s exceeded
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.469      0.364      0.362      0.209
                     pedes        548       8844      0.513      0.424      0.444        0.2
                    people        548       5125      0.463      0.321      0.319       0.12
                   bicycle        548       1287      0.278      0.178      0.143     0.0558
                       car        548      14064      0.661      0.739      0.755      0.522
                       van        548       1975      0.463      0.404      0.395      0.276
                     truck        548        750      0.535      0.353      0.361      0.234
                  tricycle        548       1045      0.471      0.211      0.223      0.121
                   awntric        548        532      0.247      0.128      0.107      0.068
                       bus        548        251      0.547       0.47      0.481      0.328
                     motor        548       4886      0.511      0.411      0.393      0.168
    Results saved to [1mruns/train/exp1_50ep[0m



```python
!ls
%cd ./yolov5
!python train.py --resume
!pkill jupyter
```


```python

! cp /kaggle/working/yolov5/runs/train/exp2_50ep2/weights/best.pt ./saves/best_exp2_50ep.py
```


```python
!python train.py --data cocoCastom.yaml --weights /kaggle/working/yolov5/runs/train/exp1_50ep/weights/last.pt --epochs 50 --cache --name exp2_50ep
```

    [Errno 2] No such file or directory: './yolov5'
    /kaggle/working/yolov5
    [34m[1mwandb[0m: WARNING ⚠️ wandb is deprecated and will be removed in a future release. See supported integrations at https://github.com/ultralytics/yolov5#integrations.
    2024-06-10 05:22:38.393931: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-06-10 05:22:38.393991: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-06-10 05:22:38.395448: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    [34m[1mwandb[0m: (1) Create a W&B account
    [34m[1mwandb[0m: (2) Use an existing W&B account
    [34m[1mwandb[0m: (3) Don't visualize my results
    [34m[1mwandb[0m: Enter your choice: (30 second timeout) 
    [34m[1mwandb[0m: W&B disabled due to login timeout.
    [34m[1mtrain: [0mweights=/kaggle/working/yolov5/runs/train/exp1_50ep/weights/last.pt, cfg=, data=cocoCastom.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=50, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, evolve_population=data/hyps, resume_evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp2_50ep, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest, ndjson_console=False, ndjson_file=False
    [34m[1mgithub: [0mup to date with https://github.com/ultralytics/yolov5 ✅
    YOLOv5 🚀 v7.0-321-g3742ab49 Python-3.10.13 torch-2.1.2 CUDA:0 (Tesla P100-PCIE-16GB, 16276MiB)
    
    [34m[1mhyperparameters: [0mlr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
    [34m[1mComet: [0mrun 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
    [34m[1mTensorBoard: [0mStart with 'tensorboard --logdir runs/train', view at http://localhost:6006/
    
                     from  n    params  module                                  arguments                     
      0                -1  1      5280  models.common.Conv                      [3, 48, 6, 2, 2]              
      1                -1  1     41664  models.common.Conv                      [48, 96, 3, 2]                
      2                -1  2     65280  models.common.C3                        [96, 96, 2]                   
      3                -1  1    166272  models.common.Conv                      [96, 192, 3, 2]               
      4                -1  4    444672  models.common.C3                        [192, 192, 4]                 
      5                -1  1    664320  models.common.Conv                      [192, 384, 3, 2]              
      6                -1  6   2512896  models.common.C3                        [384, 384, 6]                 
      7                -1  1   2655744  models.common.Conv                      [384, 768, 3, 2]              
      8                -1  2   4134912  models.common.C3                        [768, 768, 2]                 
      9                -1  1   1476864  models.common.SPPF                      [768, 768, 5]                 
     10                -1  1    295680  models.common.Conv                      [768, 384, 1, 1]              
     11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     12           [-1, 6]  1         0  models.common.Concat                    [1]                           
     13                -1  2   1182720  models.common.C3                        [768, 384, 2, False]          
     14                -1  1     74112  models.common.Conv                      [384, 192, 1, 1]              
     15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     16           [-1, 4]  1         0  models.common.Concat                    [1]                           
     17                -1  2    296448  models.common.C3                        [384, 192, 2, False]          
     18                -1  1    332160  models.common.Conv                      [192, 192, 3, 2]              
     19          [-1, 14]  1         0  models.common.Concat                    [1]                           
     20                -1  2   1035264  models.common.C3                        [384, 384, 2, False]          
     21                -1  1   1327872  models.common.Conv                      [384, 384, 3, 2]              
     22          [-1, 10]  1         0  models.common.Concat                    [1]                           
     23                -1  2   4134912  models.common.C3                        [768, 768, 2, False]          
     24      [17, 20, 23]  1     60615  models.yolo.Detect                      [10, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [192, 384, 768]]
    Model summary: 291 layers, 20907687 parameters, 20907687 gradients, 48.3 GFLOPs
    
    Transferred 481/481 items from /kaggle/working/yolov5/runs/train/exp1_50ep/weights/last.pt
    [34m[1mAMP: [0mchecks passed ✅
    [34m[1moptimizer:[0m SGD(lr=0.01) with parameter groups 79 weight(decay=0.0), 82 weight(decay=0.0005), 82 bias
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
    /opt/conda/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()
    [34m[1mtrain: [0mScanning /kaggle/input/visdrone2019/labels/train... 6471 images, 0 backgr[0m
    [34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/visdrone2019/images/train/0000137_02220_d_0000163.jpg: 1 duplicate labels removed
    [34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/visdrone2019/images/train/0000140_00118_d_0000002.jpg: 1 duplicate labels removed
    [34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/visdrone2019/images/train/9999945_00000_d_0000114.jpg: 1 duplicate labels removed
    [34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/visdrone2019/images/train/9999987_00000_d_0000049.jpg: 1 duplicate labels removed
    [34m[1mtrain: [0mWARNING ⚠️ Cache directory /kaggle/input/visdrone2019/labels is not writeable: [Errno 30] Read-only file system: '/kaggle/input/visdrone2019/labels/train.cache.npy'
    [34m[1mtrain: [0mCaching images (4.9GB ram): 100%|██████████| 6471/6471 [00:28<00:00, 226.[0m
    [34m[1mval: [0mScanning /kaggle/input/visdrone2019/labels/val... 548 images, 0 backgrounds[0m
    [34m[1mval: [0mWARNING ⚠️ Cache directory /kaggle/input/visdrone2019/labels is not writeable: [Errno 30] Read-only file system: '/kaggle/input/visdrone2019/labels/val.cache.npy'
    [34m[1mval: [0mCaching images (0.4GB ram): 100%|██████████| 548/548 [00:03<00:00, 171.76it[0m
    
    [34m[1mAutoAnchor: [0m5.73 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
    Plotting labels to runs/train/exp2_50ep2/labels.jpg... 
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    Image sizes 640 train, 640 val
    Using 4 dataloader workers
    Logging results to [1mruns/train/exp2_50ep2[0m
    Starting training for 50 epochs...
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           0/49      7.11G    0.08999     0.1439    0.02135        293        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.472      0.377      0.374      0.213
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           1/49      7.11G     0.0913     0.1464    0.02171        641        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.474      0.362      0.365      0.204
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           2/49      7.12G    0.09275     0.1515    0.02251        483        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.451      0.345      0.336      0.182
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           3/49      7.12G    0.09372     0.1554    0.02344        820        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.458      0.343      0.344      0.189
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           4/49      7.12G    0.09372     0.1545    0.02357        675        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.452      0.343      0.341      0.185
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           5/49      7.12G    0.09383     0.1558    0.02354        402        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.442      0.355      0.348      0.189
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           6/49      7.12G     0.0937      0.153    0.02337        481        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.459      0.351       0.35      0.192
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           7/49      7.12G    0.09354     0.1543    0.02349        579        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.451      0.355      0.349      0.188
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           8/49      7.12G    0.09296     0.1532    0.02334        775        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.452      0.353      0.348       0.19
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           9/49      7.12G    0.09299     0.1529    0.02313        449        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.459      0.357      0.352      0.196
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          10/49      7.12G    0.09258     0.1517    0.02294        326        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.462      0.351      0.345      0.189
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          11/49      7.12G    0.09251      0.151    0.02294        260        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.462      0.347      0.351      0.194
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          12/49      7.12G    0.09225     0.1511    0.02261        643        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.469      0.358      0.355      0.198
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          13/49      7.12G    0.09221     0.1518    0.02267        612        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.458      0.352      0.351      0.197
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          14/49      7.12G    0.09201     0.1495     0.0225        670        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.471      0.354      0.356      0.197
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          15/49      7.12G      0.092     0.1507    0.02237        375        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.454       0.36      0.354      0.197
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          16/49      7.12G    0.09183     0.1494    0.02227        555        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.463       0.36      0.363      0.203
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          17/49      7.12G    0.09145      0.148    0.02217        575        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.457      0.366      0.362      0.203
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          18/49      7.12G    0.09188     0.1491    0.02211        710        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.472      0.353      0.359      0.201
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          19/49      7.12G    0.09134     0.1507      0.022        770        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.463      0.364       0.36      0.201
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          20/49      7.12G    0.09134     0.1481    0.02178        572        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.477      0.362      0.362      0.203
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          21/49      7.12G    0.09116     0.1478    0.02187        621        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.472      0.362      0.364      0.205
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          22/49      7.12G    0.09102     0.1471    0.02172        717        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.465      0.366      0.361      0.204
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          23/49      7.12G     0.0907     0.1459    0.02171        661        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.476      0.362      0.365      0.207
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          24/49      7.12G    0.09077     0.1468    0.02154        316        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.472      0.371      0.366      0.207
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          25/49      7.12G    0.09079     0.1467    0.02154        715        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.479      0.369      0.367      0.207
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          26/49      7.12G     0.0905      0.147    0.02125        684        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.471       0.37      0.366      0.208
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          27/49      7.12G    0.09043     0.1443    0.02111        622        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.486      0.369      0.373      0.211
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          28/49      7.12G    0.09003     0.1456     0.0211        436        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.473      0.372      0.366      0.207
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          29/49      7.12G    0.09007      0.145    0.02118        624        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.486      0.366      0.368      0.209
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          30/49      7.12G     0.0902     0.1446    0.02095        425        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.479      0.374      0.371       0.21
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          31/49      7.12G    0.08985     0.1436    0.02085        471        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.467      0.373      0.368      0.209
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          32/49      7.12G    0.08987     0.1446    0.02067        752        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.476      0.376      0.373      0.213
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          33/49      7.12G    0.08985     0.1426    0.02085        379        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.483      0.371      0.373      0.213
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          34/49      7.12G    0.08985     0.1442    0.02069        617        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.475      0.379      0.373      0.212
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          35/49      7.12G    0.08931     0.1428    0.02058        536        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.488       0.37      0.377      0.215
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          36/49      7.12G    0.09002      0.142    0.02049        401        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.482      0.376      0.374      0.213
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          37/49      7.12G    0.08917     0.1438    0.02038        691        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.478       0.38      0.374      0.215
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          38/49      7.12G    0.08933     0.1417    0.02034        574        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.479      0.371      0.372      0.212
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          39/49      7.12G    0.08901     0.1409    0.02027        750        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.496      0.371      0.375      0.215
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          40/49      7.12G     0.0896     0.1406    0.02035        841        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.49      0.368      0.373      0.214
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          41/49      7.12G    0.08874     0.1397    0.02009        576        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.483      0.374      0.376      0.216
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          42/49      7.12G    0.08924     0.1426    0.02002        589        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.49      0.374      0.377      0.217
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          43/49      7.12G      0.089     0.1376    0.01981        315        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.485      0.376      0.378      0.218
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          44/49      7.12G    0.08871     0.1397    0.01985        792        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.49      0.376       0.38       0.22
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          45/49      7.12G    0.08849     0.1394    0.01976        651        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.49      0.375      0.379      0.219
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          46/49      7.12G    0.08903     0.1389     0.0198        336        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.491      0.377       0.38      0.219
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          47/49      7.12G    0.08844       0.14    0.01968        384        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.485      0.376       0.38      0.219
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          48/49      7.12G     0.0883      0.139    0.01965        500        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.49      0.375      0.381       0.22
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          49/49      7.12G    0.08797      0.136    0.01953        628        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.478      0.383      0.381       0.22
    
    50 epochs completed in 2.782 hours.
    Optimizer stripped from runs/train/exp2_50ep2/weights/last.pt, 42.2MB
    Optimizer stripped from runs/train/exp2_50ep2/weights/best.pt, 42.2MB
    
    Validating runs/train/exp2_50ep2/weights/best.pt...
    Fusing layers... 
    Model summary: 212 layers, 20889303 parameters, 0 gradients, 48.0 GFLOPs
                     Class     Images  Instances          P          R      mAP50   WARNING ⚠️ NMS time limit 2.100s exceeded
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.473      0.376      0.373      0.217
                     pedes        548       8844      0.517      0.429      0.449      0.203
                    people        548       5125      0.481      0.339      0.341      0.127
                   bicycle        548       1287        0.3      0.189      0.154       0.06
                       car        548      14064      0.679      0.742      0.762      0.529
                       van        548       1975      0.453      0.415        0.4      0.281
                     truck        548        750      0.517      0.359      0.371      0.251
                  tricycle        548       1045      0.408      0.255      0.217      0.116
                   awntric        548        532       0.25      0.147      0.125     0.0807
                       bus        548        251      0.601      0.458      0.503      0.345
                     motor        548       4886      0.521      0.428      0.408      0.174
    Results saved to [1mruns/train/exp2_50ep2[0m



```python
!rm -rf /kaggle/working/yolov5/runs/train/exp2_50ep
!rm -rf /kaggle/working/yolov5_v1_out1.zip
%cd ..
! zip -r yolov5_v1_out1.zip ./yolov5 
```


```python
!python detect.py --weights /kaggle/working/yolov5/runs/train/exp2_50ep2/weights/best.pt --conf 0.25 --source "https://www.youtube.com/watch?v=MNn9qKG2UFI&pp=ygUZdHJhZmZpYyByb2FkIGNhbWVyYSBtb3ZpZQ%3D%3D"
```

    [34m[1mdetect: [0mweights=['/kaggle/working/yolov5/runs/train/exp2_50ep2/weights/best.pt'], source=https://www.youtube.com/watch?v=MNn9qKG2UFI&pp=ygUZdHJhZmZpYyByb2FkIGNhbWVyYSBtb3ZpZQ%3D%3D, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_csv=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1
    YOLOv5 🚀 v7.0-321-g3742ab49 Python-3.10.13 torch-2.1.2 CUDA:0 (Tesla P100-PCIE-16GB, 16276MiB)
    
    Fusing layers... 
    Model summary: 212 layers, 20889303 parameters, 0 gradients, 48.0 GFLOPs
    WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()
    
    1/1: https://www.youtube.com/watch?v=MNn9qKG2UFI&pp=ygUZdHJhZmZpYyByb2FkIGNhbWVyYSBtb3ZpZQ%3D%3D...  Success (9184 frames 1280x720 at 30.00 FPS)
    
    Traceback (most recent call last):
      File "/kaggle/working/yolov5/detect.py", line 312, in <module>
        main(opt)
      File "/kaggle/working/yolov5/detect.py", line 307, in main
        run(**vars(opt))
      File "/opt/conda/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
        return func(*args, **kwargs)
      File "/kaggle/working/yolov5/detect.py", line 134, in run
        for path, im, im0s, vid_cap, s in dataset:
      File "/kaggle/working/yolov5/utils/dataloaders.py", line 505, in __next__
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord("q"):  # q to quit
    cv2.error: OpenCV(4.10.0) /io/opencv/modules/highgui/src/window.cpp:1367: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvWaitKey'
    
    terminate called without an active exception



```python
visdrone_yaml = """
path : /kaggle/input/visdrone2019
train : /kaggle/input/visdrone2019/images/train
val : /kaggle/input/visdrone2019/images/val 
names:

  0: pedes #pedestrian

  1: people

  2: bicycle

  3: car

  4: van

  5: truck

  6: tricycle

  7: awntric #awning-tricycle

  8: bus

  9: motor
download: |
  from utils.general import download, Path


  # Download labels
  segments = False  # segment or box labels
  dir = Path(yaml['path'])  # dataset root dir
  url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
  urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels
  download(urls, dir=dir.parent)

  # Download data
  urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
          'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
          'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
  download(urls, dir=dir / 'images', threads=3)

"""

# Save the changes to a new YAML file
with open('/kaggle/working/yolov5/data/cocoCastom.yaml', 'w') as file:
    file.write(visdrone_yaml)

```


```python
!python val.py --weights /kaggle/working/yolov5/runs/train/exp3_60ep/weights/best.pt --data cocoCastom.yaml --img 640 --half --verbose
```

    [Errno 2] No such file or directory: './yolov5'
    /kaggle/working/yolov5
    [34m[1mval: [0mdata=/kaggle/working/yolov5/data/cocoCastom.yaml, weights=['/kaggle/working/yolov5/runs/train/exp3_60ep/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=True, save_txt=False, save_hybrid=False, save_conf=False, save_json=False, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
    YOLOv5 🚀 v7.0-321-g3742ab49 Python-3.10.13 torch-2.1.2 CUDA:0 (Tesla P100-PCIE-16GB, 16276MiB)
    
    Fusing layers... 
    Model summary: 212 layers, 20889303 parameters, 0 gradients, 48.0 GFLOPs
    Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
    100%|████████████████████████████████████████| 755k/755k [00:00<00:00, 68.7MB/s]
    [34m[1mval: [0mScanning /kaggle/input/visdrone2019/labels/val... 548 images, 0 backgrounds[0m
    [34m[1mval: [0mWARNING ⚠️ Cache directory /kaggle/input/visdrone2019/labels is not writeable: [Errno 30] Read-only file system: '/kaggle/input/visdrone2019/labels/val.cache.npy'
                     Class     Images  Instances          P          R      mAP50   WARNING ⚠️ NMS time limit 2.100s exceeded
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.504       0.36      0.369      0.216
                     pedes        548       8844      0.559      0.414      0.447      0.204
                    people        548       5125      0.508      0.309      0.328      0.124
                   bicycle        548       1287      0.326      0.176       0.16     0.0622
                       car        548      14064      0.702       0.73      0.759      0.531
                       van        548       1975      0.477        0.4      0.392      0.279
                     truck        548        750      0.559       0.35      0.363      0.241
                  tricycle        548       1045      0.419      0.227      0.221      0.118
                   awntric        548        532      0.272      0.139      0.124     0.0797
                       bus        548        251      0.669      0.458      0.506      0.349
                     motor        548       4886      0.549      0.393      0.394       0.17
    Speed: 0.1ms pre-process, 4.6ms inference, 14.2ms NMS per image at shape (32, 3, 640, 640)
    Results saved to [1mruns/val/exp4[0m



```python
!python detect.py --weights /kaggle/working/yolov5/runs/train/exp2_50ep2/weights/best.pt --conf 0.25 --source /kaggle/input/testvideo/4K\ Road\ traffic\ video\ for\ object\ detection\ and\ tracking\ -\ free\ download\ now.mp4
```


```python
!python detect.py --weights /kaggle/working/yolov5/runs/train/exp3_60ep/weights/best.pt --conf 0.25 --source /kaggle/input/testvideo/4K\ Road\ traffic\ video\ for\ object\ detection\ and\ tracking\ -\ free\ download\ now.mp4
```


```python
!pkill jupyter
```

    ^C



```python
!ls
%cd ./yolov5
```

    state.db  yolov5  yolov5_v1_out1.zip
    /kaggle/working/yolov5



```python
!mv /kaggle/input/visdrone-dataset/VisDrone2019-DET-val/VisDrone2019-DET-val/* /kaggle/input/visdrone-dataset/VisDrone2019-DET-val/
```

    mv: cannot move '/kaggle/input/visdrone-dataset/VisDrone2019-DET-val/VisDrone2019-DET-val/annotations' to '/kaggle/input/visdrone-dataset/VisDrone2019-DET-val/annotations': Read-only file system
    mv: cannot move '/kaggle/input/visdrone-dataset/VisDrone2019-DET-val/VisDrone2019-DET-val/images' to '/kaggle/input/visdrone-dataset/VisDrone2019-DET-val/images': Read-only file system



```python
visdrone_yaml = """
# Ultralytics YOLO 🚀, AGPL-3.0 license

# VisDrone2019-DET dataset https://github.com/VisDrone/VisDrone-Dataset by Tianjin University

# Example usage: yolo train data=VisDrone.yaml

# parent

# ├── ultralytics

# └── datasets

#     └── VisDrone  ← downloads here (2.3 GB)# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]



path: D:\detection\visdrone  # dataset root dir

train: VisDrone2019-DET-train/images  # train images (relative to 'path')  6471 images

val: VisDrone2019-DET-val/images  # val images (relative to 'path')  548 images

test: VisDrone2019-DET-test_dev/images  # test images (optional)  1610 images# Classes

# Classes
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor

"""

# Save the changes to a new YAML file
with open('/kaggle/working/yolov5/data/cocoCastom_DS2.yaml', 'w') as file:
    file.write(visdrone_yaml)
```


```python
!rm -rf /kaggle/working/yolov5/runs/train/exp3_60ep*
```


```python
!python train.py --data cocoCastom.yaml --weights /kaggle/working/yolov5/runs/train/exp2_50ep2/weights/last.pt --epochs 60 --cache --name exp3_60ep
```

    [Errno 2] No such file or directory: './yolov5'
    /kaggle/working/yolov5
    [34m[1mwandb[0m: WARNING ⚠️ wandb is deprecated and will be removed in a future release. See supported integrations at https://github.com/ultralytics/yolov5#integrations.
    2024-06-11 08:06:05.148417: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-06-11 08:06:05.148489: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-06-11 08:06:05.150059: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    [34m[1mwandb[0m: (1) Create a W&B account
    [34m[1mwandb[0m: (2) Use an existing W&B account
    [34m[1mwandb[0m: (3) Don't visualize my results
    [34m[1mwandb[0m: Enter your choice: (30 second timeout) 
    [34m[1mwandb[0m: W&B disabled due to login timeout.
    [34m[1mtrain: [0mweights=/kaggle/working/yolov5/runs/train/exp2_50ep2/weights/last.pt, cfg=, data=cocoCastom.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=60, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, evolve_population=data/hyps, resume_evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp3_60ep, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest, ndjson_console=False, ndjson_file=False
    [34m[1mgithub: [0mup to date with https://github.com/ultralytics/yolov5 ✅
    YOLOv5 🚀 v7.0-321-g3742ab49 Python-3.10.13 torch-2.1.2 CUDA:0 (Tesla P100-PCIE-16GB, 16276MiB)
    
    [34m[1mhyperparameters: [0mlr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
    [34m[1mComet: [0mrun 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
    [34m[1mTensorBoard: [0mStart with 'tensorboard --logdir runs/train', view at http://localhost:6006/
    
                     from  n    params  module                                  arguments                     
      0                -1  1      5280  models.common.Conv                      [3, 48, 6, 2, 2]              
      1                -1  1     41664  models.common.Conv                      [48, 96, 3, 2]                
      2                -1  2     65280  models.common.C3                        [96, 96, 2]                   
      3                -1  1    166272  models.common.Conv                      [96, 192, 3, 2]               
      4                -1  4    444672  models.common.C3                        [192, 192, 4]                 
      5                -1  1    664320  models.common.Conv                      [192, 384, 3, 2]              
      6                -1  6   2512896  models.common.C3                        [384, 384, 6]                 
      7                -1  1   2655744  models.common.Conv                      [384, 768, 3, 2]              
      8                -1  2   4134912  models.common.C3                        [768, 768, 2]                 
      9                -1  1   1476864  models.common.SPPF                      [768, 768, 5]                 
     10                -1  1    295680  models.common.Conv                      [768, 384, 1, 1]              
     11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     12           [-1, 6]  1         0  models.common.Concat                    [1]                           
     13                -1  2   1182720  models.common.C3                        [768, 384, 2, False]          
     14                -1  1     74112  models.common.Conv                      [384, 192, 1, 1]              
     15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     16           [-1, 4]  1         0  models.common.Concat                    [1]                           
     17                -1  2    296448  models.common.C3                        [384, 192, 2, False]          
     18                -1  1    332160  models.common.Conv                      [192, 192, 3, 2]              
     19          [-1, 14]  1         0  models.common.Concat                    [1]                           
     20                -1  2   1035264  models.common.C3                        [384, 384, 2, False]          
     21                -1  1   1327872  models.common.Conv                      [384, 384, 3, 2]              
     22          [-1, 10]  1         0  models.common.Concat                    [1]                           
     23                -1  2   4134912  models.common.C3                        [768, 768, 2, False]          
     24      [17, 20, 23]  1     60615  models.yolo.Detect                      [10, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [192, 384, 768]]
    Model summary: 291 layers, 20907687 parameters, 20907687 gradients, 48.3 GFLOPs
    
    Transferred 481/481 items from /kaggle/working/yolov5/runs/train/exp2_50ep2/weights/last.pt
    [34m[1mAMP: [0mchecks passed ✅
    [34m[1moptimizer:[0m SGD(lr=0.01) with parameter groups 79 weight(decay=0.0), 82 weight(decay=0.0005), 82 bias
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
    /opt/conda/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()
    [34m[1mtrain: [0mScanning /kaggle/input/visdrone2019/labels/train... 6471 images, 0 backgr[0m
    [34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/visdrone2019/images/train/0000137_02220_d_0000163.jpg: 1 duplicate labels removed
    [34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/visdrone2019/images/train/0000140_00118_d_0000002.jpg: 1 duplicate labels removed
    [34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/visdrone2019/images/train/9999945_00000_d_0000114.jpg: 1 duplicate labels removed
    [34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/visdrone2019/images/train/9999987_00000_d_0000049.jpg: 1 duplicate labels removed
    [34m[1mtrain: [0mWARNING ⚠️ Cache directory /kaggle/input/visdrone2019/labels is not writeable: [Errno 30] Read-only file system: '/kaggle/input/visdrone2019/labels/train.cache.npy'
    [34m[1mtrain: [0mCaching images (4.9GB ram): 100%|██████████| 6471/6471 [00:28<00:00, 227.[0m
    [34m[1mval: [0mScanning /kaggle/input/visdrone2019/labels/val... 548 images, 0 backgrounds[0m
    [34m[1mval: [0mWARNING ⚠️ Cache directory /kaggle/input/visdrone2019/labels is not writeable: [Errno 30] Read-only file system: '/kaggle/input/visdrone2019/labels/val.cache.npy'
    [34m[1mval: [0mCaching images (0.4GB ram): 100%|██████████| 548/548 [00:03<00:00, 176.14it[0m
    
    [34m[1mAutoAnchor: [0m5.73 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
    Plotting labels to runs/train/exp3_60ep/labels.jpg... 
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    Image sizes 640 train, 640 val
    Using 4 dataloader workers
    Logging results to [1mruns/train/exp3_60ep[0m
    Starting training for 60 epochs...
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           0/59      7.11G    0.08841     0.1387    0.01961        293        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.489      0.372      0.377      0.216
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           1/59      7.11G     0.0895     0.1403     0.0198        641        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.485      0.369       0.37       0.21
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           2/59      7.12G    0.09083     0.1452    0.02057        483        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.464      0.366      0.354      0.194
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           3/59      7.12G    0.09191     0.1496    0.02164        820        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.478      0.361       0.36      0.201
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           4/59      7.12G    0.09195     0.1492    0.02182        675        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.469      0.353      0.352      0.193
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           5/59      7.12G    0.09221     0.1509    0.02192        402        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.449      0.355      0.347      0.191
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           6/59      7.12G    0.09217     0.1485    0.02183        481        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.459      0.357       0.35      0.193
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           7/59      7.12G    0.09211     0.1499    0.02197        579        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.468      0.356      0.356      0.197
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           8/59      7.12G    0.09155      0.149    0.02186        775        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.456      0.357      0.351      0.196
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           9/59      7.12G    0.09166     0.1488    0.02172        449        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.459      0.367      0.359      0.199
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          10/59      7.12G    0.09127     0.1475    0.02156        326        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.464      0.359      0.354      0.195
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          11/59      7.12G    0.09131      0.147    0.02159        260        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.475      0.353      0.357        0.2
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          12/59      7.12G    0.09106     0.1474    0.02131        643        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.475      0.363      0.363      0.202
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          13/59      7.12G    0.09097      0.148    0.02136        612        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.474      0.364      0.362      0.202
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          14/59      7.12G     0.0909     0.1459    0.02125        670        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.475      0.359      0.355      0.197
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          15/59      7.12G    0.09092     0.1472    0.02119        375        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.472       0.36       0.36      0.203
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          16/59      7.12G    0.09082     0.1458    0.02112        555        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.468      0.367      0.363      0.205
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          17/59      7.12G    0.09044     0.1445      0.021        575        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.471      0.362      0.359      0.203
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          18/59      7.12G    0.09086     0.1457    0.02095        710        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.493      0.355       0.36      0.202
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          19/59      7.12G    0.09042     0.1476    0.02086        770        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.474      0.361      0.362      0.204
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          20/59      7.12G    0.09042     0.1448    0.02067        572        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.481      0.358      0.362      0.204
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          21/59      7.12G     0.0903     0.1448     0.0208        621        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.467      0.372      0.367      0.207
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          22/59      7.12G    0.09013     0.1441    0.02062        717        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.48      0.359      0.363      0.206
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          23/59      7.12G    0.08984      0.143    0.02063        661        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.478      0.362       0.36      0.205
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          24/59      7.12G    0.08993     0.1438    0.02053        316        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.472      0.367      0.361      0.206
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          25/59      7.12G       0.09     0.1439    0.02056        715        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.48      0.371      0.369      0.209
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          26/59      7.12G    0.08973     0.1443    0.02025        684        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.484      0.365      0.367       0.21
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          27/59      7.12G    0.08972     0.1419    0.02015        622        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.49      0.367      0.373      0.212
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          28/59      7.12G    0.08928     0.1433    0.02016        436        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.488      0.368      0.369      0.208
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          29/59      7.12G    0.08935     0.1427    0.02023        624        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.492      0.364      0.369       0.21
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          30/59      7.12G    0.08953     0.1425    0.02004        425        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.489      0.373      0.373      0.212
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          31/59      7.12G    0.08921     0.1414    0.01994        471        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.488       0.37      0.373      0.212
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          32/59      7.12G    0.08923     0.1425     0.0198        752        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.49      0.372      0.374      0.213
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          33/59      7.12G    0.08924     0.1407    0.02001        379        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759        0.5      0.371      0.378      0.214
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          34/59      7.12G    0.08928     0.1424    0.01986        617        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.491      0.372      0.375      0.213
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          35/59      7.12G    0.08875      0.141    0.01978        536        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.501      0.376      0.379      0.216
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          36/59      7.12G    0.08947     0.1404    0.01971        401        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.489      0.366      0.372      0.213
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          37/59      7.12G    0.08862     0.1423     0.0196        691        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.496      0.366      0.375      0.216
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          38/59      7.12G    0.08881     0.1404    0.01956        574        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.506      0.366      0.374      0.214
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          39/59      7.12G    0.08852     0.1397    0.01951        750        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.502      0.374      0.379      0.217
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          40/59      7.12G    0.08914     0.1392    0.01958        841        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.497      0.368      0.375      0.214
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          41/59      7.12G     0.0883     0.1386    0.01936        576        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.491       0.37      0.377      0.217
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          42/59      7.12G    0.08881     0.1415    0.01933        589        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.506      0.368      0.377      0.217
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          43/59      7.12G    0.08858     0.1366    0.01914        315        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.501       0.37      0.378      0.217
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          44/59      7.12G    0.08827     0.1388    0.01914        792        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.498      0.375      0.379      0.218
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          45/59      7.12G    0.08809     0.1385    0.01908        651        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.493      0.379      0.377      0.217
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          46/59      7.12G    0.08864      0.138    0.01912        336        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.497       0.38      0.379      0.217
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          47/59      7.12G    0.08806     0.1392    0.01902        384        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.503      0.372      0.378      0.218
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          48/59      7.12G    0.08793     0.1383      0.019        500        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.497      0.376       0.38      0.218
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          49/59      7.12G    0.08762     0.1353    0.01883        628        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.511      0.368      0.379      0.218
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          50/59      7.12G    0.08787     0.1374    0.01892        350        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.499      0.374      0.382       0.22
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          51/59      7.12G     0.0879     0.1381    0.01866        465        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.498      0.371      0.379      0.219
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          52/59      7.12G    0.08789     0.1379    0.01871        542        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.497      0.374       0.38      0.219
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          53/59      7.12G    0.08842     0.1382    0.01874        480        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.509      0.373       0.38      0.219
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          54/59      7.12G    0.08772     0.1359     0.0186        658        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.513       0.37       0.38      0.221
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          55/59      7.12G    0.08729     0.1365    0.01855        295        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.505      0.371      0.381      0.221
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          56/59      7.12G    0.08781     0.1369    0.01852        455        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.511       0.37      0.382       0.22
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          57/59      7.12G     0.0875     0.1369    0.01849        898        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759       0.51      0.372      0.382      0.221
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          58/59      7.12G    0.08758     0.1367     0.0184        502        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.508      0.373      0.383      0.222
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          59/59      7.12G    0.08756     0.1357    0.01832        656        640: 1
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.507      0.372      0.381      0.221
    
    60 epochs completed in 3.332 hours.
    Optimizer stripped from runs/train/exp3_60ep/weights/last.pt, 42.2MB
    Optimizer stripped from runs/train/exp3_60ep/weights/best.pt, 42.2MB
    
    Validating runs/train/exp3_60ep/weights/best.pt...
    Fusing layers... 
    Model summary: 212 layers, 20889303 parameters, 0 gradients, 48.0 GFLOPs
                     Class     Images  Instances          P          R      mAP50   WARNING ⚠️ NMS time limit 2.100s exceeded
                     Class     Images  Instances          P          R      mAP50   
                       all        548      38759      0.504      0.366      0.374      0.218
                     pedes        548       8844      0.558      0.416      0.449      0.205
                    people        548       5125      0.507      0.321      0.338      0.127
                   bicycle        548       1287      0.328      0.179      0.161     0.0628
                       car        548      14064      0.702      0.734      0.763      0.533
                       van        548       1975      0.477      0.402      0.394       0.28
                     truck        548        750      0.556      0.353      0.365      0.243
                  tricycle        548       1045       0.42      0.237      0.228      0.122
                   awntric        548        532      0.278      0.147      0.127     0.0823
                       bus        548        251      0.667      0.462      0.512      0.353
                     motor        548       4886      0.552      0.407      0.406      0.175
    Results saved to [1mruns/train/exp3_60ep[0m



```python
! zip -r /kaggle/working/yolov5_v1_out2.zip /kaggle/working/yolov5 
```
