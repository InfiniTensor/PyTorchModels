# HuggingFace 模型

此目录下的脚本支持用于进行问答的 HuggingFace 模型在 SQuAD v2.0 数据集上的训练和推理任务，具体包括下列模型：

- **Albert**: `AlbertForQuestionAnswering`
- **Bart**: `BartForQuestionAnswering`
- **Bert**: `BertForQuestionAnswering`
- **BigBird**: `BigBirdForQuestionAnswering`
- **BigBirdPegasus**: `BigBirdPegasusForQuestionAnswering`
- **Bloom**: `BloomForQuestionAnswering`
- **Camembert**: `CamembertForQuestionAnswering`
- **Canine**: `CanineForQuestionAnswering`
- **ConvBert**: `ConvBertForQuestionAnswering`
- **Data2VecText**: `Data2VecTextForQuestionAnswering`
- **Deberta**: `DebertaForQuestionAnswering`
- **DebertaV2**: `DebertaV2ForQuestionAnswering`
- **DistilBert**: `DistilBertForQuestionAnswering`
- **Electra**: `ElectraForQuestionAnswering`
- **Ernie**: `ErnieForQuestionAnswering`
- **ErnieM**: `ErnieMForQuestionAnswering`
- **Falcon**: `FalconForQuestionAnswering`
- **Flaubert**: `FlaubertForQuestionAnsweringSimple`
- **FNet**: `FNetForQuestionAnswering`
- **Funnel**: `FunnelForQuestionAnswering`
- **GPT2**: `GPT2ForQuestionAnswering`
- **GPTNeo**: `GPTNeoForQuestionAnswering`
- **GPTNeoX**: `GPTNeoXForQuestionAnswering`
- **GPTJ**: `GPTJForQuestionAnswering`
- **IBert**: `IBertForQuestionAnswering`
- **LayoutLMv2**: `LayoutLMv2ForQuestionAnswering`
- **LayoutLMv3**: `LayoutLMv3ForQuestionAnswering`
- **LED**: `LEDForQuestionAnswering`
- **Lilt**: `LiltForQuestionAnswering`
- **Llama**: `LlamaForQuestionAnswering`
- **Longformer**: `LongformerForQuestionAnswering`
- **Luke**: `LukeForQuestionAnswering`
- **Lxmert**: `LxmertForQuestionAnswering`
- **MarkupLM**: `MarkupLMForQuestionAnswering`
- **MBart**: `MBartForQuestionAnswering`
- **Mega**: `MegaForQuestionAnswering`
- **MegatronBert**: `MegatronBertForQuestionAnswering`
- **MobileBert**: `MobileBertForQuestionAnswering`
- **MPNet**: `MPNetForQuestionAnswering`
- **Mpt**: `MptForQuestionAnswering`
- **Mra**: `MraForQuestionAnswering`
- **MT5**: `MT5ForQuestionAnswering`
- **Mvp**: `MvpForQuestionAnswering`
- **Nezha**: `NezhaForQuestionAnswering`
- **Nystromformer**: `NystromformerForQuestionAnswering`
- **OPT**: `OPTForQuestionAnswering`
- **QDQBert**: `QDQBertForQuestionAnswering`
- **Reformer**: `ReformerForQuestionAnswering`
- **RemBert**: `RemBertForQuestionAnswering`
- **Roberta**: `RobertaForQuestionAnswering`
- **RobertaPreLayerNorm**: `RobertaPreLayerNormForQuestionAnswering`
- **RoCBert**: `RoCBertForQuestionAnswering`
- **RoFormer**: `RoFormerForQuestionAnswering`
- **Splinter**: `SplinterForQuestionAnswering`
- **SqueezeBert**: `SqueezeBertForQuestionAnswering`
- **T5**: `T5ForQuestionAnswering`
- **UMT5**: `UMT5ForQuestionAnswering`
- **XLM**: `XLMForQuestionAnsweringSimple`
- **XLMRoberta**: `XLMRobertaForQuestionAnswering`
- **XLMRobertaXL**: `XLMRobertaXLForQuestionAnswering`
- **XLNet**: `XLNetForQuestionAnsweringSimple`
- **Xmod**: `XmodForQuestionAnswering`
- **Yoso**: `YosoForQuestionAnswering`

## 环境要求
- `pip install -r requirements.txt`
  
### 有互联网连接时
- 使用互联网连接 HuggingFaceHub 时，需要更换国内镜像源：
  - `export HF_ENDPOINT=https://hf-mirror.com`

### 无互联网连接时
- 使用预训练模型时，需要提前**手动下载模型文件**。以 `bert-base-uncased` 为例，到[官方 repo](https://hf-mirror.com/google-bert/bert-base-uncased/tree/main)下载如下几个文件，并将其放到同一目录下，例如 `./bert-base-uncased`：
```
  ./bert-base-uncased/
  ├── config.json
  ├── pytorch_model.bin
  ├── tokenizer_config.json
  ├── tokenizer.json
  └── vocab.txt
```
- 需要手动下载[SQuAD v2.0 数据集](https://rajpurkar.github.io/SQuAD-explorer/)放在指定同一目录下，目录结构为：
```
   squad/
   ├── dev-v2.0.json
   └── train-v2.0.json
```
- 其他无互联网连接时才需要的前置条件（**已经配齐，不需要做额外收集**）：
  - 需要本地调用 `evaluate` 库中 `metrics` 模块的[代码](https://github.com/huggingface/evaluate/tree/main/metrics)，主要需要的是与 SQuAD 数据集相关的两个目录，已经一同放在 `./metrics/` 目录下。
  - 需要本地调用 SQuAD v2.0 数据集的[处理代码](https://hf-mirror.com/datasets/rajpurkar/squad_v2/tree/main)，已经一同放在 `./squad_v2.py`。
  
## 多卡分布式训练

使用 `--do_train` 进行训练，使用 `--per_device_train_batch_size N` 指定训练过程中使用的 batch size。

脚本会默认使用可见的所有加速卡进行计算，建议使用 `export CUDA_VISIBLE_DEVICES=0,1,2,3` 来显式指定要使用的加速卡序号，可以参考如下的实现。

### 单机多卡

#### 有互联网连接时

有互联网连接时，只需要更换国内镜像源，指定模型名称和数据集名称后会自动进行下载预训练模型。可以参考 `run_train_online.sh` 的实现。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
    --nproc_per_node=4 \
    qa.py \
    --model_name_or_path [model_name(e.g. bert-base-uncased)] \
    --dataset_name squad_v2 \
    --version_2_with_negative \
    --per_device_train_batch_size 10 \
    --do_train \
    --output_dir /tmp/debug_squad/ \
```

#### 无互联网连接时

无互联网连接时，需要指定模型文件路径和数据集路径。假设已提前将上述 `bert-base-uncased` 模型文件下载好并存放在 `./bert-base-uncased/` 目录下。可以参考 `run_train_offline.sh` 的实现。

```bash
export CUDA_VISIBLE_DEVICES=0
export SQUAD_PATH=./squad

MODEL_PATH=./bert-base-uncased

torchrun \
    --nproc_per_node=4 \
    qa.py \
    --model_name_or_path $MODEL_PATH \
    --config_name $MODEL_PATH \
    --tokenizer_name $MODEL_PATH \
    --train_file $SQUAD_PATH/train-v2.0.json \
    --validation_file $SQUAD_PATH/dev-v2.0.json \
    --test_file $SQUAD_PATH/dev-v2.0.json \
    --version_2_with_negative \
    --per_device_train_batch_size 10 \
    --do_eval \
    --output_dir /tmp/debug_squad/ \
```

### 多机多卡

与上述运行命令类似，只需要修改 torchrun 的启动参数即可。

Node 0：

```bash
torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=$IP_OF_NODE0 \
  --master_port=$FREEPORT \
  qa.py
```

Node 1：

```bash
torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=$IP_OF_NODE0 \
  --master_port=$FREEPORT \
  qa.py
```

## 推理

使用 `--do_eval` 进行推理，使用 `--per_device_eval_batch_size N` 指定推理过程中使用的 batch size。其余参数与上述命令类似。

#### 有互联网连接时

可以参考 `run_eval_online.sh` 的实现。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

torchrun \
    --nproc_per_node=1 \
    qa.py \
    --model_name_or_path [model_name(e.g. bert-base-uncased)] \
    --dataset_name squad_v2 \
    --version_2_with_negative \
    --per_device_eval_batch_size 10 \
    --do_eval \
    --output_dir /tmp/debug_squad/ \
```

#### 无互联网连接时

假设已提前将 `bert-base-uncased` 模型文件下载好并存放在 `./bert-base-uncased/` 目录下。可以参考 `run_eval_offline.sh` 的实现。

```bash
export CUDA_VISIBLE_DEVICES=0
export SQUAD_PATH=./squad

MODEL_PATH=./bert-base-uncased

torchrun \
    --nproc_per_node=1 \
    qa.py \
    --model_name_or_path $MODEL_PATH \
    --config_name $MODEL_PATH \
    --tokenizer_name $MODEL_PATH \
    --train_file $SQUAD_PATH/train-v2.0.json \
    --validation_file $SQUAD_PATH/dev-v2.0.json \
    --test_file $SQUAD_PATH/dev-v2.0.json \
    --version_2_with_negative \
    --do_eval \
    --output_dir /tmp/debug_squad/ \
```

