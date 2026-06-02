# 测试命令手册

所有命令都需要先加载环境：`cd /data/shared/baoming/workplace/PyTorchModels && source env.sh`

## 一、IC 模型测试（4个模型）

```bash
bash run_all.sh all ImageClassification
```

当前模型列表：resnet18、mobilenet_v2、vgg16、vit_b_16

## 二、其他模型测试

```bash
# 全部（Detection + NLP + Speech + RL + Recommendation + SR）
bash run_all.sh all Detection NLP Speech RL Recommendation SR

# 单独测某个域
bash run_all.sh all Detection          # fasterrcnn + ssd + yolo
bash run_all.sh all NLP                # bert
bash run_all.sh all Speech             # deepspeech2 + wav2vec
bash run_all.sh all RL                 # dqn
bash run_all.sh all Recommendation     # dlrm
bash run_all.sh all SR                 # espcn
```

## 三、全量回归测试（IC 4个 + 其他全部）

```bash
bash run_all.sh all
```

## 四、单独调试某个模型

```bash
# fasterrcnn
cd Detection/fasterrcnn && DATA_DIR=../data/VOCdevkit bash run_train.sh 2>&1 | tee train.log
cd Detection/fasterrcnn && DATA_DIR=../data/VOCdevkit bash run_eval.sh 2>&1 | tee eval.log

# yolo eval
cd Detection/yolo && bash run_eval.sh 2>&1 | tee eval.log

# deepspeech2
cd Speech/deepspeech2 && bash run_train.sh 2>&1 | tee train.log
cd Speech/deepspeech2 && bash run_eval.sh 2>&1 | tee eval.log

# wav2vec
cd Speech/wav2vec && bash run_train_online.sh 2>&1 | tee train.log
cd Speech/wav2vec && bash run_eval_online.sh 2>&1 | tee eval.log

# dlrm
cd Recommendation/DLRM && bash run_train.sh 2>&1 | tee train.log

# bert train
cd NLP/HuggingFace && bash run_train_online.sh 2>&1 | tee train.log
```

## 五、修改 IC 模型数量

编辑 `run_all.sh` 第 58-60 行的 `IC_MODELS` 数组：

```bash
# 当前：4个模型
IC_MODELS=(
    resnet18 mobilenet_v2 vgg16 vit_b_16
)

# 恢复全量：取消注释第 62-77 行，删除上面的 4 个
```
