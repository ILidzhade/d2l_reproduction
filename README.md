# Leakage-controlled Dream2Learn

This project independently reproduces the central mechanism of
[Dream2Learn](https://doi.org/10.1007/s11263-026-02958-1) while replacing its
pretrained diffusion model with a conditional variational autoencoder trained
only on observed Split Mini-ImageNet data. It compares replay, random latent
dreams, and directed latent dreams using the same improved CVAE initialization.

## Setup

1. Install Conda or Miniconda.
2. Create the required environment:

   ```bash
   conda env create -f environment.yml
   ```

## Obtain Mini-ImageNet

Mini-ImageNet is derived from ImageNet and is not included in this repository.
Request access and download `ILSVRC2012_img_train.tar` from the
[official ImageNet site](https://image-net.org/download-images), subject to its
terms. Then use the canonical
[mini-imagenet-tools](https://github.com/yaoyao-liu/mini-imagenet-tools)
generator and Ravi--Larochelle CSV splits:

```bash
git clone https://github.com/yaoyao-liu/mini-imagenet-tools.git ../mini-imagenet-tools
cd ../mini-imagenet-tools
conda run -n dream2learn python mini_imagenet_generator.py \
  --tar_dir /absolute/path/to/ILSVRC2012_img_train.tar
cd -
mkdir -p data/images
cp ../mini-imagenet-tools/csv_files/{train,val,test}.csv data/
find ../mini-imagenet-tools/processed_images -type f -name '*.jpg' \
  -exec cp {} data/images/ +
```

The resulting layout must be:

```text
data/
├── images/
├── train.csv
├── val.csv
└── test.csv
```

Each CSV must contain `filename,label` columns. The generator creates the
100-class, 600-image-per-class, 84×84 dataset used by this reproduction.

## Reproduce the submission

From a clean checkout with the dataset in `data/`, this single resumable command
trains the 100-epoch balanced CVAE base, runs all three continual-learning
conditions, and regenerates every submitted experimental number and figure:

```bash
conda run -n dream2learn python reproduce.py
```

Results are written to
`outputs/phase8_final_balanced_mechanism_seed-42.json`, and the report figure is
written to `docs/report/final-balanced-results.png`. Interrupted continual runs
resume from their latest task boundary.

## Code map

- `src/models.py`: legacy and balanced conditional VAEs.
- `src/data.py`: dataset partitioning and continual task stream.
- `src/replay.py`: bounded reservoir replay.
- `src/dreaming.py`: dream optimization, assignment, and reuse.
- `src/training.py`: real, replay, reconstruction, KL, and dream losses.
- `src/evaluation.py`: final accuracy, forgetting, and forward transfer.
- `src/experiment.py`: experiment lifecycle and resumable checkpoints.
- `reproduce.py`: complete submission reproduction entry point.
- `main.ipynb`: interactive version of the final comparison.

Run the unit tests with:

```bash
conda run -n dream2learn python -m unittest tests.test_d2l
```
