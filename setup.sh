# conda create -n qwen-eval-env python=3.10 -y
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate qwen-eval-env
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
# conda install tqdm
# pip install pycocotools
# conda install transformers==4.57.0
# pip install qwen_vl_utils
# pip install accelerate
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl


#!/bin/bash
# Exit immediately if a command fails
set -e

api_key=$1

echo "=== Creating conda environment ==="
conda create -n detpo-env python=3.10 -y

echo "=== Activating environment ==="
# Conda activate doesn't work directly in non-interactive shells unless you source it
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate detpo-env
echo "Active Conda environment: $CONDA_DEFAULT_ENV"


echo "===installing vllm in fp8 env===="
pip install vllm

echo "=== Installing flashinfer ==="
# pip install flashinfer --upgrade
# conda install -c conda-forge gcc=11 gxx=11 -y
pip install flashinfer-python


echo "=== Installing tqdm ==="
conda install -y tqdm

echo "=== Installing pycocotools ==="
pip install pycocotools

echo "=== Installing transformers ==="
pip install transformers==4.57.0

echo "=== Installing qwen_vl_utils ==="
pip install qwen_vl_utils==0.0.8

echo "=== Installing accelerate ==="
pip install accelerate

echo "=== Installing opencv ==="
conda install -c conda-forge opencv -y

echo "=== Installing supervision ==="
conda install -c conda-forge supervision -y

echo "===upgrading numpy==="
conda install -c conda-forge numpy=2.0.1 -y

echo "=== installing torch-c-dlpack-ext ===" # for vllm fp8 support
pip install torch-c-dlpack-ext

echo "=== installing roboflow ==="
conda deactivate

conda create -n rf100vl-env python=3.9.23 -y
echo "=== Activating environment ==="
# Conda activate doesn't work directly in non-interactive shells unless you source it
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rf100vl-env
echo "Active Conda environment: $CONDA_DEFAULT_ENV"

pip install rf100vl==1.1.0

echo "===exporting roboflow key==="
export ROBOFLOW_API_KEY=$api_key

echo "=== Downloading the data ==="
mkdir -p datasets/rf100-vl-fsod
python download_data_roboflow.py --data_dir datasets/rf100-vl-fsod

echo "===making dir for result==="
mkdir -p results

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate detpo-env
echo "Active Conda environment: $CONDA_DEFAULT_ENV"

echo "===script successful!!! ======"