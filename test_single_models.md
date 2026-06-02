# 单独模型测试命令

按顺序执行，每个几分钟就能看到结果。报错后把 `xxx.log` 最后几行发给调试。

## 1. Detection - fasterrcnn (train FAIL + eval FAIL)

```bash
# train（会生成 checkpoint 给 eval 用）
cd /data/shared/baoming/workplace/PyTorchModels/Detection/fasterrcnn && source ../../env.sh && DATA_DIR=../data/VOCdevkit bash run_train.sh 2>&1 | tee train.log

# eval（等 train 跑完后再执行）
cd /data/shared/baoming/workplace/PyTorchModels/Detection/fasterrcnn && source ../../env.sh && DATA_DIR=../data/VOCdevkit bash run_eval.sh 2>&1 | tee eval.log
```

## 2. Detection - yolo (eval FAIL)

```bash
cd /data/shared/baoming/workplace/PyTorchModels/Detection/yolo && source ../../env.sh && bash run_eval.sh 2>&1 | tee eval.log
```

## 3. Speech - deepspeech2 (train FAIL + eval FAIL)

```bash
# train
cd /data/shared/baoming/workplace/PyTorchModels/Speech/deepspeech2 && source ../../env.sh && bash run_train.sh 2>&1 | tee train.log

# eval
cd /data/shared/baoming/workplace/PyTorchModels/Speech/deepspeech2 && source ../../env.sh && bash run_eval.sh 2>&1 | tee eval.log
```

## 4. Speech - wav2vec (train FAIL + eval FAIL)

```bash
# train
cd /data/shared/baoming/workplace/PyTorchModels/Speech/wav2vec && source ../../env.sh && bash run_train_online.sh 2>&1 | tee train.log

# eval
cd /data/shared/baoming/workplace/PyTorchModels/Speech/wav2vec && source ../../env.sh && bash run_eval_online.sh 2>&1 | tee eval.log
```

## 5. Recommendation - dlrm (train FAIL)

```bash
cd /data/shared/baoming/workplace/PyTorchModels/Recommendation/DLRM && source ../../env.sh && bash run_train.sh 2>&1 | tee train.log
```

## 6. NLP - bert (train throughput=N/A)

```bash
cd /data/shared/baoming/workplace/PyTorchModels/NLP/HuggingFace && source ../../env.sh && bash run_train_online.sh 2>&1 | tee train.log
```

## 已通过的模型（不需要重测）

- Detection/ssd: train OK, eval OK
- RL/dqn: train OK, eval OK
- Recommendation/dlrm: eval OK
- SR/espcn: train OK, eval OK
- NLP/bert: eval OK

## 全部修好后的完整回归命令

```bash
cd /data/shared/baoming/workplace/PyTorchModels && source env.sh && bash run_all.sh all Detection NLP Speech RL Recommendation SR
```
