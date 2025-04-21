#!/bin/bash

# 定义要处理的文件夹列表
folders=("deeplab" "fcn" "lraspp" "unet")

# 遍历每个文件夹
for folder in "${folders[@]}"; do
    # 检查文件夹是否存在
    if [ -d "$folder" ]; then
        echo "进入文件夹: $folder"
        cd "$folder" || exit 1  # 如果进入失败则退出

        # 检查 run_train.sh 是否存在
        if [ -f "run_train.sh" ]; then
            echo "正在运行 run_train.sh，日志输出到 ${folder}_train.log"
            # 运行脚本并重定向输出到日志文件
            bash ./run_train.sh > "../${folder}_train.log" 2>&1
            echo "完成 $folder 的训练"
        else
            echo "错误：$folder 中没有找到 run_train.sh"
        fi

        # 返回上级目录
        cd ..
    else
        echo "错误：文件夹 $folder 不存在"
    fi
done

echo "所有训练任务已完成"
