SHELL := /bin/bash

VENV ?= .venv-packaging
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip

.PHONY: help venv install-cpu install-gpu-cuda118 intel-xpu run-local run-local-watch

help:
	@echo "Mothbot development targets"
	@echo ""
	@echo "Quick start:"
	@echo "  make -f make/dev.mk install-cpu           # default setup"
	@echo "  make -f make/dev.mk run-local             # run local Gradio"
	@echo "  make -f make/dev.mk run-local-watch       # run local Gradio with auto-restart"
	@echo ""
	@echo "Optional:"
	@echo "  make -f make/dev.mk venv"
	@echo "  make -f make/dev.mk install-gpu-cuda118   # optional NVIDIA CUDA machines"

venv:
	python3 -m venv "$(VENV)"
	@echo "Created $(VENV)"

install-cpu: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[cpu,packaging]"

run-local:
	$(PYTHON) -m ui.local_main

run-local-watch:
	$(PYTHON) -m ui.local_watch

install-gpu-cuda118: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[cuda118,packaging]" --extra-index-url https://download.pytorch.org/whl/cu118

install-xpu: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[xpu,packaging]" --extra-index-url https://download.pytorch.org/whl/xpu
	@$(PIP) install onnxruntime-openvino \
		&& echo "  onnxruntime-openvino installed — OpenVINOExecutionProvider available (Intel GPU/NPU acceleration)." \
		|| echo "  onnxruntime-openvino has no wheel for this Python version — staying on plain onnxruntime (CPU-only ONNX Runtime). torch.xpu still accelerates the main YOLO detection path."
