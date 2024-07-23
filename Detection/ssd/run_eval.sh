
#!/bin/bash

# 这个脚本用于运行Python脚本
# 数据集放在同级目录下  /data1/shared/Dataset/VOCdevkit 

# 确保脚本在遇到错误时停止执行
set -e

export CUDA_VISIBLE_DEVICES=0

# 检查软连接是否已经存在了
if [ -e "../data/VOCdevkit" ]; then
    echo "../data/VOCdevkit exists"
else
    # 创建软连接
    ln -s  /data1/shared/Dataset/VOCdevkit ../data
fi

ckpt_path="./checkpoint_ssd300.pth.tar"

ckpt_url="https://cloud.tsinghua.edu.cn/seafhttp/files/9d78bd21-6e46-4677-b825-f03af94cec00/checkpoint_ssd300.pth.tar"

# execute create_data_lists.py
echo "executing create_data_lists.py..."
python create_data_lists.py

if [ -e $ckpt_path ]; then
    echo "$ckpt_path exists"
else
    echo "Download $ckpt_path from url $ckpt_url"
    # Download model ckpt from server
    wget $ckpt_url
fi

# execute detect.py
echo "Evaluate SSD START"
python eval.py

# rm $ckpt_path

echo "Evaluate SSD FINISHED"


# mAP 0.771