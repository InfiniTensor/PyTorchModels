## Quick Start Guide
#model training
```bash
bash run_scripts/SSD_VGG16_FP32_10000S_4MLUs_Train.sh 100 64 2e-3
```
#model inference
```bash
bash run_scripts/SSD_VGG16_Infer.sh run_scripts/ssd_final.pth /dataset/VOC2007/
```
