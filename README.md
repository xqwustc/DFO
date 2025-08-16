
# DFO
Implementation of "Mitigating Redundancy in Deep Recommender Systems: A Field Importance Distribution Perspective," published at `KDD 2025` [[Paper](https://dl.acm.org/doi/10.1145/3690624.3709275)].

## Clone FuxiCTR and Use Our Code
Clone the [FuxiCTR](https://github.com/reczoo/FuxiCTR) repository to your local machine with the following command:
```
git clone https://github.com/reczoo/FuxiCTR.git
```

Then, copy the code from our repository into the `FuxiCTR/model_zoo/` directory.

## Dataset Preparation
For Frappe, Avazu (We use the Avazu_x4_001 version), and iFly-AD, you can directly download them from [FuxiCTR-Datasets](https://github.com/reczoo/Datasets/tree/main). Detailed download and configuration instructions are provided there.

For the Ali-CCP dataset, the process is slightly more complicated as we need to split part of the test set into a validation set (since the original data does not provide a validation set). First, download the dataset from https://tianchi.aliyun.com/dataset/408, and then run the following code:
```
python ./data/AliCCP/ali-ccp/preprocess_ali_ccp.py
```

This will generate `ali_ccp_train/val/test.csv`.

For the above four datasets, please place the downloaded data in the `FuxiCTR/data` folder. You can refer to the corresponding dataset folder names in the `FuxiCTR/model_zoo/DNN/DNN_torch/config/dataset_config.yaml` file.

## Pretraining with Learners
The following examples use the WideDeep model. Other models can be adapted similarly and are fully compatible with all models in FuxiCTR (hyperparameter configurations are provided in the config files).
```
python FuxiCTR/model_zoo/WideDeep/WideDeep_torch/run_expid_dfo_select.py --gpu xxx --expid WideDeep_[frappe/aliccp/avazu/iflychu]
```
For example, if you want to run the experiment on the `Avazu` dataset, set the `expid` to `WideDeep_avazu` (the same applies below).

This will generate a `feature_importance_result.csv` file, sorted by `feature_weight` in descending order.

## DFO without Adaptive Embedding Sizes (w/o AE)
```
python FuxiCTR/model_zoo/WideDeep/WideDeep_torch/run_expid_dfo_wo_ae.py --gpu xxx --expid WideDeep_[frappe/aliccp/avazu/iflychu] --K [xx]
```
Here, `--K` specifies the number of feature fields to retain, which can be a list of integers.

## DFO with Adaptive Embedding Sizes (w/ AE)
```
python FuxiCTR/model_zoo/WideDeep/WideDeep_torch/run_expid_dfo_w_ae.py --gpu xxx --expid WideDeep_[frappe/aliccp/avazu/iflychu] --K [xx]
```
Similarly, `--K` specifies the number of feature fields to retain, but this will also trigger embedding size assignment.

Results can be viewed either through the console or log files.

### Contact Us & Acknowledgments
If you have any questions, feel free to contact Xianquan Wang (email: wxqcn@mail.ustc.edu.cn). Our work heavily relies on the excellent contributions of [FuxiCTR](https://github.com/reczoo/FuxiCTR). We sincerely thank the team for their efforts.

Cite us if our work helps you~
```bibtex
@inproceedings{wang2025mitigating,
  title={Mitigating redundancy in deep recommender systems: A field importance distribution perspective},
  author={Wang, Xianquan and Wu, Likang and Li, Zhi and Yuan, Haitao and Shen, Shuanghong and Xu, Huibo and Su, Yu and Lei, Chenyi},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1},
  pages={1515--1526},
  year={2025}
}
```
