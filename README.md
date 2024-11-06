# RecSys2023-SINE
The official code repository of our RecSys 2023 paper.

Yunzhu Pan, Chen Gao, Jianxin Chang, Yanan Niu, Yang Song, Kun Gai, Depeng Jin, Yong Li,

Understanding and Modeling Passive-Negative Feedback for Short-video Sequential Recommendation, 

17th ACM Conference on Recommender Systems.

## Requirements
* Linux
* Python 3.8+
* PyTorch 1.9+
* Numpy
* Pandas
* Tqdm
* torchmetrics

## Dataset

Download from [WeChat-Big-Data-Challenge-2021](https://github.com/WeChat-Big-Data-Challenge-2021/WeChat_Big_Data_Challenge).

## Model Training

Simply run the following command to reproduce the experiments on corresponding dataset and model:
```
python main.py --mode train --task wechat --behavior_path {dataset path} --work_dir train --maxlen 200 --gpu_id 0 --num_interest 2 --adaptive --metrics auc gauc ndcg_2
```

For parameters such as learning rate, batch size, and early stopping, please refer to `parameters.py`

## Evaluate on test dataset

Change `--mode` and `--work_dir` to `test` and add `--state_dict_path {checkpoint path}`. Other parameters should remain the same as in train mode.

```
python main.py --mode test --task wechat --behavior_path {dataset path} --work_dir test --state_dict_path {checkpoint path} --maxlen 200  --gpu_id 0 --num_interest 2 --adaptive --metrics auc gauc ndcg_2
```

## Citation
If you use our codes and datasets in your research, please cite:
```
@inproceedings{pan2023understanding,
  title={Understanding and modeling passive-negative feedback for short-video sequential recommendation},
  author={Pan, Yunzhu and Gao, Chen and Chang, Jianxin and Niu, Yanan and Song, Yang and Gai, Kun and Jin, Depeng and Li, Yong},
  booktitle={Proceedings of the 17th ACM conference on recommender systems},
  pages={540--550},
  year={2023}
}
```

