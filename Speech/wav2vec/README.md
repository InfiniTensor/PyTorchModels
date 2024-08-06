# Wav2Vec 模型

此目录下的脚本支持用于进行语音识别的 Wav2Vec 系列模型在 LibriSpeech 数据集上的训练和推理任务，具体包括下列模型：

- [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large)
- [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus)
- [facebook/wav2vec2-large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60)
- [facebook/hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k)
- [asapp/sew-mid-100k](https://huggingface.co/asapp/sew-mid-100k)

## 环境要求
- `pip install -r requirements.txt`
- 手动下载[LibriSpeech 数据集](https://www.openslr.org/12/)的 clean 版本。由于整个数据集文件过大，代码实现会默认从本地加载下载好的数据集，而不会调用 huggingface 提供的 datasets 库中相关用于在线下载数据集的接口。
  ```bash
  wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
  wget http://www.openslr.org/resources/12/dev-clean.tar.gz
  wget http://www.openslr.org/resources/12/test-clean.tar.gz

  tar -xzvf train-clean-100.tar.gz -C /path/to/local/dataset
  tar -xzvf dev-clean.tar.gz -C /path/to/local/dataset
  tar -xzvf test-clean.tar.gz -C /path/to/local/dataset
  ```
  数据集解压好后，目录结构为：
  ```
   /path/to/local/dataset/Librispeech/
   ├── BOOKS.TXT
   ├── CHAPTERS.TXT
   ├── LICENSE.TXT
   ├── README.TXT
   ├── SPEAKERS.TXT
   ├── dev-clean/
   │   ├── 1272/
   │   ├── ...
   ├── test-clean/
   │   ├── 1272/
   │   ├── ...
   └── train-clean-100/
       ├──1272/
       ├── ...
	```
  运行脚本中必须显示指定 `LIBRISPEECH_PATH` 环境变量为上述数据集的路径，代码中会读取该环境变量加载数据。
- 运行常见报错（来自 `evaluate` 库）：
  ```
  ImportError: /${YOUR_PYTHON_PATH}/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
  ___________________________________________________________________________
  Contents of /${YOUR_PYTHON_PATH}/site-packages/sklearn/__check_build:
  __init__.py               _check_build.cpython-36m-aarch64-linux-gnu.so__pycache__
  setup.py
  ___________________________________________________________________________
  It seems that scikit-learn has not been built correctly.
  ```
  这种情况下需要在环境变量中添加：`export LD_PRELOAD=$LD_PRELOAD:/PATH/TO/libgomp-d22c30c5.so.1.0.0`
  
### 有互联网连接时
- 使用互联网连接 HuggingFaceHub 时，需要更换国内镜像源：
  - `export HF_ENDPOINT=https://hf-mirror.com`

### 无互联网连接时
- 需要本地调用 evaluate 库中 metrics 模块的代码，主要需要的是与 WER(Word Error Rate) 指标相关的代码，已经放在 ./metrics/ 目录下（**已经配齐，不需要做额外收集**）。
- 使用预训练模型时，需要提前**手动下载模型文件**。以 `wav2vec2-large-lv60` 为例，到[官方 repo](https://hf-mirror.com/facebook/wav2vec2-large-lv60)下载如下几个文件，并将其放到同一目录下，例如 `./wav2vec2-large-lv60`：
```
  ./wav2vec2-large-lv60/
  ├── config.json
  ├── pytorch_model.bin
  ├── preprocessor_config.json
  └── vocab.json
```
