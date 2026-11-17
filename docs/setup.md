# Setup Options

Run these commands from the repository root after cloning with submodules.

## Docker Builds

```bash
make build            # CUDA 12.4
make build-blackwell  # CUDA 12.8
make build-cpu        # CPU
```

The Docker build uses the pinned `contact_graspnet_pytorch` submodule.
For an existing checkout, fetch it before the build:

```bash
git submodule update --init --recursive
```

## Docker Display Access

Before a Docker GUI command, permit the current local user to access the X server:

```bash
xhost +si:localuser:$(id -un)
```

## CPU Run

```bash
make inference IMAGE=cgn_pytorch-cpu GPU_FLAGS=
make visualize IMAGE=cgn_pytorch-cpu GPU_FLAGS=
```

Inference and visualization open graphical windows. Docker GUI commands require a Linux desktop with X11 access.

## Host Visualization

Use Python 3.10 or 3.11 with a local graphical display.

```bash
make setup-venv
cgn_visualize_venv/bin/python -m p1.visualize_predictions \
  --npz results/predictions_9.npz --topk 50 --out results/viz_9.png
```

This environment only loads saved predictions. It does not run model inference.

## Model Configuration

Inference loads the pretrained checkpoint and its configuration from
`/opt/cgn/checkpoints/contact_graspnet/` inside the Docker image.
The root `configs/config.yaml` is a reference copy and does not override that configuration.
