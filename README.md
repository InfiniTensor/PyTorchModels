# PyTorchModels

## 项目结构
```
PyTorchModels \
	- ImageClassification \
	- NLP \
	- Detection \
	- Segmentation
```

## How to use this repo in Ascend/Cambricon hardware platform?

Test is performed in docker for this two platform. 

### Important: Before test, you need to link the right dataset paths to our repo!

### Ascend platform
1. init Ascend toolkit by `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
2. Replace the right paltform name in env script and init platform env variable by `source env.sh`
3. Add our `usercustomize.py` path to PYTHONPATH by `export PYTHONPATH=$PYTHONPATH:{YOUR_REPO_PATH}`
4. All is ready! Let's test our models training in Ascend platform by the following command:
    ```
    nohup bash run_train_all.sh >> output.log 2>&1 &
    ```

### Cambricon platform
1. Please create a sitecustomize.py file in your python site-packages folder, and add the following code:
    ```
    import site
    site.ENABLE_USER_SITE = True
    ```
2. Replace the right paltform name in env script and and init platform env variable by `source env.sh`
3. Add our `usercustomize.py` path to PYTHONPATH by `export PYTHONPATH=$PYTHONPATH:{YOUR_REPO_PATH}`
4. All is ready! Let's test our models training in Ascend platform by the following command:
    ```
    nohup bash run_train_all.sh >> output.log 2>&1 &
    ``` 
