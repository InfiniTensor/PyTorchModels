# 配置环境
```
pip install -r requirements.txt
```

# 数据集准备
```
mkdir datasets
cd datasets
ln -s /data1/shared/Dataset/coco/ ./
```

# 训练
```
python3 train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5s.yaml  --batch-size 1
```

# 验证
```
python3 val.py --weights yolov5s.pt --data coco.yaml --img 640
```

# 推理
```
python3 detect.py --weights yolov5s.pt --source data/images/zidane.jpg
```