#!/bin/bash

# 项目来源：https://github.com/dxyang/DQN_pytorch

# 运行方式
for i in 0 1 2 3 4 5 6; do
	python main.py train --task-id $i
done
