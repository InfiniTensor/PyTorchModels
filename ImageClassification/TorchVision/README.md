# TorchVision 模型

此目录下的脚本支持用于进行图像分类的 TorchVision 模型在 ImageNet 数据集上的训练和推理任务，具体包括下列模型的若干规模：

- **AlexNet**: `alexnet`
- **ConvNeXt**: `convnext_base`, `convnext_large`, `convnext_small`, `convnext_tiny`
- **DenseNet**: `densenet121`, `densenet161`, `densenet169`, `densenet201`,
- **EfficientNet**: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`, `efficientnet_b4`, `efficientnet_b5`, `efficientnet_b6`, `efficientnet_b7`, `efficientnet_v2_l`, `efficientnet_v2_m`, `efficientnet_v2_s`
- **GoogleNet**: `googlenet`
- **Inception**: `inception_v3`
- **MnasNet**: `mnasnet0_5`, `mnasnet0_75`, `mnasnet1_0`, `mnasnet1_3`
- **MobileNet**: `mobilenet_v2`, `mobilenet_v3_large`, `mobilenet_v3_small`
- **RegNet**: `regnet_x_16gf`, `regnet_x_1_6gf`, `regnet_x_32gf`, `regnet_x_3_2gf`, `regnet_x_400mf`, `regnet_x_800mf`, `regnet_x_8gf`, `regnet_y_128gf`, `regnet_y_16gf`, `regnet_y_1_6gf`, `regnet_y_32gf`, `regnet_y_3_2gf`, `regnet_y_400mf`, `regnet_y_800mf`, `regnet_y_8gf`
- **ResNet**: `resnet101`, `resnet152`, `resnet18`, `resnet34`, `resnet50`
- **ResNeXt**: `resnext101_32x8d`, `resnext101_64x4d`, `resnext50_32x4d`
- **ShuffleNet**: `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0`
- **SqueezeNet**: `squeezenet1_0`, `squeezenet1_1`
- **VGG**: `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`
- **ViT**: `vit_b_16`, `vit_b_32`, `vit_h_14`, `vit_l_16`, `vit_l_32`
- **WideResNet**: `wide_resnet101_2`, `wide_resnet50_2`

# 模型实际运行情况
模型在 python==3.10.12 torch==1.11.0+cu113 torchvision==0.12.0+cu113 环境下的运行的实际情况如下表

| Models            |      Train Loss         |       Eval Acc   |
|------------------ | --------------------|----------------|
| alexnet           |                     |        56.550      |
| convnext_base |
| convnext_large |
| convnext_small |
| convnext_tiny |
| densenet121 |
| densenet161 |
| densenet169 |
| densenet201 |
| efficientnet_b0 |
| efficientnet_b1 |
| efficientnet_b2 |
| efficientnet_b3 |
| efficientnet_b4 |
| efficientnet_b5 |
| efficientnet_b6 |
| efficientnet_b7 |
| googlenet |       | 69.772|
| inception_v3 |    
| mnasnet0_5 |
| mnasnet0_75 |
| mnasnet1_0 |
| mnasnet1_3 |
| mobilenet_v2 |                        |   71.868 |
| mobilenet_v3_large |
| mobilenet_v3_small |
| regnet_x_16gf |
| regnet_x_1_6gf |
| regnet_x_32gf |
| regnet_x_3_2gf |
| regnet_x_400mf |
| regnet_x_800mf |
| regnet_x_8gf |
| regnet_y_128gf |
| regnet_y_16gf |
| regnet_y_1_6gf |
| regnet_y_32gf |
| regnet_y_3_2gf |
| regnet_y_400mf |
| regnet_y_800mf |
| regnet_y_8gf |
| resnet101 |
| resnet152 |
| resnet18 |
| resnet34 |
| resnet50 |   |                 76.148   |
| resnext101_32x8d |
| resnext50_32x4d |
| shufflenet_v2_x0_5 |
| shufflenet_v2_x1_0 |
| shufflenet_v2_x1_5 |
| shufflenet_v2_x2_0 |
| squeezenet1_0 |
| squeezenet1_1 |
| vgg11 |
| vgg11_bn |
| vgg13 |
| vgg13_bn |
| vgg16 |                      | 71.580   |
| vgg16_bn |
| vgg19 |                      | 72.390   |
| vgg19_bn |
| vit_b_16 |
| vit_b_32 |
| vit_l_16 |
| vit_l_32 |
| wide_resnet101_2 |
| wide_resnet50_2 |
 
## 环境要求

- `pip install -r requirements.txt`
- ImageNet 数据集，文件结构为：
  ```bash
    imagenet/train/
    ├── n01440764
    │   ├── n01440764_10026.JPEG
    │   ├── n01440764_10027.JPEG
    │   ├── ......
    ├── ......
    imagenet/val/
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ......
    ├── ......
  ```
  > ImageNet 数据集构建方式：
  >
  > 1. 从[官方网站](http://www.image-net.org/)下载数据集，可以得到两个`.tar`文件：
  > - ILSVRC2012_img_train.tar (about 138 GB)
  > - ILSVRC2012_img_val.tar (about 6.3 GB)
  >
  > 2. 使用[该脚本](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh)进行提取。
  > 

## 多卡分布式训练

脚本会默认使用可见的所有加速卡进行计算，建议使用 `export CUDA_VISIBLE_DEVICES=0,1,2,3` 来显式指定要使用的加速卡序号，可以参考 `run_train.sh` 的实现。

### 单机多卡

```bash
python main.py 
    -a resnet50 
    --dist-url `tcp://127.0.0.1:FREEPORT` 
    --dist-backend `nccl` 
    --multiprocessing-distributed 
    --world-size 1 
    --rank 0 
    [imagenet-folder with train and val folders]
```

### 多机多卡

Node 0:

```bash
python main.py 
    -a resnet50 
    --dist-url `tcp://IP_OF_NODE0:FREEPORT` 
    --dist-backend `nccl` 
    --multiprocessing-distributed 
    --world-size 2 
    --rank 0 
    [imagenet-folder with train and val folders]
```

Node 1:

```bash
python main.py 
    -a resnet50 
    --dist-url `tcp://IP_OF_NODE0:FREEPORT` 
    --dist-backend `nccl` 
    --multiprocessing-distributed 
    --world-size 2 
    --rank 1 
    [imagenet-folder with train and val folders]
```

## 推理

脚本会默认使用可见的所有加速卡进行计算，建议使用 `export CUDA_VISIBLE_DEVICES=0,1,2,3` 来显式指定要使用的加速卡序号，可以参考 `run_eval.sh` 的实现。

使用 `--evaluate` 来进行推理，使用 `--pretrained` 来通过加载官方提供的预训练权重构建模型（需要连接互联网），使用 `--weights WEIGHTS_PATH` 来指定 `.pth` 权重文件进行模型初始化。

```bash
python main.py
    -a resnet50
    --dist-backend 'nccl'
    --dist-url `tcp://127.0.0.1:FREEPORT`
    --multiprocessing-distributed
    --world-size 1
    --rank 0
    --batch-size 64
    --weights [.pth file path]
    --evaluate
    [imagenet-folder with train and val folders]
```

## 计时方法

**TODO**

## 脚本参数解释

```bash
usage: main.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR] [--momentum M] [--wd W] [-p N] [--resume PATH] [-e] [--pretrained] [--weights WEIGHTS_PATH] [--world-size WORLD_SIZE] [--rank RANK]
               [--dist-url DIST_URL] [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU] [--multiprocessing-distributed] [--dummy]
               [DIR]

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset (default: imagenet)

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: alexnet | convnext_base | convnext_large | convnext_small | convnext_tiny | densenet121 | densenet161 | densenet169 | densenet201 | efficientnet_b0 |
                        efficientnet_b1 | efficientnet_b2 | efficientnet_b3 | efficientnet_b4 | efficientnet_b5 | efficientnet_b6 | efficientnet_b7 | googlenet | inception_v3 | mnasnet0_5 | mnasnet0_75 |
                        mnasnet1_0 | mnasnet1_3 | mobilenet_v2 | mobilenet_v3_large | mobilenet_v3_small | regnet_x_16gf | regnet_x_1_6gf | regnet_x_32gf | regnet_x_3_2gf | regnet_x_400mf | regnet_x_800mf |
                        regnet_x_8gf | regnet_y_128gf | regnet_y_16gf | regnet_y_1_6gf | regnet_y_32gf | regnet_y_3_2gf | regnet_y_400mf | regnet_y_800mf | regnet_y_8gf | resnet101 | resnet152 | resnet18 |
                        resnet34 | resnet50 | resnext101_32x8d | resnext50_32x4d | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn | vit_b_16 | vit_b_32 | vit_l_16 | vit_l_32 | wide_resnet101_2 | wide_resnet50_2 (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --weights             use customized weights to initialize model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel
                        training
  --dummy               use fake data to benchmark

```
