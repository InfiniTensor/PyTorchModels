# DLRM 模型

此目录下的脚本支持用于进行推荐的 DLRM 模型在 MovieLens 数据集上的训练和推理任务。
 
## 环境要求

- `pip install -r requirements.txt`
- MovieLens 数据集，并扩展为 4x users 和 16x items。由于数据集[预处理](https://gitee.com/cambricon/pytorch_modelzoo/tree/master/built-in/recommendation/DLRM/models/data_generation/fractal_graph_expansions#step-to-expand-the-dataset-x16-users-x32-items)程序繁琐且耗时较长，建议直接使用已经完成预处理的数据集。预处理好的文件结构为：
  ```bash
    ml-20mx4x16/
    ├── alias_tbl_4x16_cached_sampler.pkl
    ├── trainx4x16_0.npz
    ├── ...
    ├── testx4x16_0.npz
    ├── ...
    ├── test_negx4x16_0.npz
    ├── ......
  ```
