# SVMT
Structure-Valid Mask Transformer

---

## Environment Setup

- Python 3.10  
- CUDA 11.8  

We provide the environment configuration for reproducibility:

```bash
conda env create -f environment.yml
conda activate hc
```

## Dataset Structure

Our dataset is organized as follows:
```bash
data/
├── data1/
│ ├── imgs/
│ │ ├── 0000.png
│ │ ├── 0001.png
│ │ ├── ...
│ ├── mask.png
│
├── data2/
│ ├── imgs/
│ │ ├── 0000.png
│ ├── mask.png
│
├── data3/
│ ├── imgs/
│ ├── mask.png
│
└── test_data/
├── imgs/
│ ├── 0000.png
│ ├── 0001.png
│ ├── ...
├── mask.png
```
test_data is a sample scene for testing and visualization.

## train 
```bash
python train.py

#The training hyperparameters are defined at the beginning of train.py.
#The model architecture and all model-specific configurations are implemented in model/SVMT.py.
```

## test 
```bash
python test.py
```
