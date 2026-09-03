# SVMT

Structure-Valid Mask Transformer for self-supervised depth and pose estimation from image sequences.

## Environment Setup

The provided environment targets Python 3.10 and PyTorch with CUDA 11.8. Run the following commands from the project root:

```bash
conda env create -f environment.yml
conda activate hc
```

The CUDA-enabled PyTorch packages are installed through the PyTorch CUDA 11.8 package index. Use a compatible NVIDIA driver on the training computer. A non-CUDA Mac can be used for static validation, while CUDA training should be run on Windows with an NVIDIA GPU.

## Dataset Structure

Training data is stored in `train_data/`. Each sequence is placed in its own directory and must contain at least three numerically named images.

```text
train_data/
├── 0001/
│   ├── imgs/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   └── mask.png
├── 0002/
│   ├── imgs/
│   └── mask.png
└── ...
```

`mask.png` is optional. When it is absent, the loader uses an all-one valid-region mask. Images within each sequence are sorted by their numeric filename prefix. Training samples are consecutive three-frame clips.

Testing data is stored in `test_data/`, with one scene per directory:

```text
test_data/
└── scene_name/
    ├── imgs/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    └── mask.png
```

Test outputs are written under each test scene directory.

## Parameters

All user-adjustable parameters are stored in `parameters_json/`:

| File | Contents |
| --- | --- |
| `train_parameters.json` | Training paths, resume checkpoint, data settings, SV mask schedule, loss weights, learning-rate schedule, augmentation, and camera intrinsics. |
| `svmt_parameters.json` | SVMT architecture and model initialization parameters. |
| `test_parameters.json` | Test paths, checkpoint path, output name, normalization, and data settings. |

`train.py` and `test.py` load these files automatically. Both scripts construct the model using `svmt_parameters.json`. Keep the image height and width in the train/test JSON files consistent with `img_size_h` and `img_size_w` in `svmt_parameters.json`.

## Training

Configure `parameters_json/train_parameters.json`, then run:

```bash
python train.py
```

Set `model_load_path` to `null` to start a new run. To resume a previous run, set it to that run's `checkpoint_latest_model.pt` path.

Each run saves two checkpoints in its timestamped directory:

- `checkpoint_best_photo_model.pt`: updated only when `mean_photo_loss` reaches a new minimum.
- `checkpoint_latest_model.pt`: updated after every epoch and includes the latest model and Adam optimizer state for resuming training.

Set `enable_interval_checkpoint_saving` to `true` in `train_parameters.json` to keep additional checkpoints. With `checkpoint_save_interval_epochs` set to `20`, the training run also saves checkpoints after epochs 20, 40, 60, and so on.

## Testing

Set `trained_depth_model_path` and `save_index` in `parameters_json/test_parameters.json`. In most cases, use `checkpoint_best_photo_model.pt` for evaluation.

```bash
python test.py
```

Depth maps and pose estimates are saved under the selected `save_index` directory inside each test scene.

## Self-Supervised Training Note

Self-supervised depth and pose optimization is stochastic during the first stage of training. Random initialization, sampled frame triplets, and photometric warping can make the first few epochs fluctuate. Evaluate the trend over several epochs rather than judging a run from its first few iterations. Once depth, pose, and valid reprojection regions become stable, the optimization is usually more consistent.
