IMAGE ?= cgn_pytorch
CONTAINER ?= cgn_run
GPU_FLAGS ?= --gpus all

RUN_FLAGS = --rm -it --user $(shell id -u):$(shell id -g) \
	-e DISPLAY=$(DISPLAY) -e XDG_RUNTIME_DIR=/tmp \
	-v /tmp/.X11-unix:/tmp/.X11-unix -v "$(CURDIR):/workspace"

build:
	docker build -t $(IMAGE) .

build-blackwell:
	docker build \
		--build-arg BASE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 \
		--build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cu128 \
		--build-arg TORCH_VERSION=2.7.0 \
		--build-arg TORCHVISION_VERSION=0.22.0 -t $(IMAGE) .

build-cpu:
	docker build --build-arg BASE=ubuntu:22.04 \
		--build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cpu \
		-t $(IMAGE)-cpu .

run:
	docker run $(RUN_FLAGS) $(GPU_FLAGS) --name $(CONTAINER) $(IMAGE)

run-cpu:
	docker run $(RUN_FLAGS) --name $(CONTAINER) $(IMAGE)-cpu

inference:
	docker run $(RUN_FLAGS) $(GPU_FLAGS) $(IMAGE) \
		python -m contact_graspnet_pytorch.inference \
		--ckpt_dir=/opt/cgn/checkpoints/contact_graspnet \
		--np_path="test_data/*.npy" --local_regions --filter_grasps

visualize:
	docker run $(RUN_FLAGS) $(GPU_FLAGS) $(IMAGE) \
		python -m p1.visualize_predictions

setup-venv:
	bash setup_venv.sh

down:
	docker stop $(CONTAINER)

clean:
	docker rmi $(IMAGE) $(IMAGE)-cpu

.PHONY: build build-blackwell build-cpu run run-cpu inference visualize setup-venv down clean
