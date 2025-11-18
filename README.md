# About DeepDTA: deep drug-target binding affinity prediction

The approach used in this work is the modeling of protein sequences and compound 1D representations (SMILES) with convolutional neural networks (CNNs) to predict the binding affinity value of drug-target pairs.

# Installation

## Data

Please see the [README](https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md) for detailed explanation.

## Env

```bash
conda env create -f deepdta.yml
```

# Usage

You can run the training, prediction, and evaluation steps individually using the scripts provided below.

Individual Steps

```bash
# 1. Train the model
bash Kd_train.sh <fold_num> <gpu_id>

# 2. Generate predictions
bash Kd_predict.sh <fold_num> <gpu_id>

# 3. Evaluate performance
bash Kd_eval.sh
```
Arguments Explanation:

`<fold_num>` (Example: 0): Represents the cross-validation fold number.

`<gpu_id>` (Example: 1): Represents the GPU ID to be used for computation (e.g., 0 or 1).

Full Pipeline
To execute the complete pipeline (training, prediction, and evaluation) in sequence, use the following command:

```bash
bash run_all_Kd.sh
```


## For citation:

```
@article{ozturk2018deepdta,
  title={DeepDTA: deep drug--target binding affinity prediction},
  author={{\"O}zt{\"u}rk, Hakime and {\"O}zg{\"u}r, Arzucan and Ozkirimli, Elif},
  journal={Bioinformatics},
  volume={34},
  number={17},
  pages={i821--i829},
  year={2018},
  publisher={Oxford University Press}
}
```
