#!/bin/bash

echo 'Date: ' `date`
echo 'Host: ' `hostname`
echo 'System: ' `uname -spo`
echo 'GPU: ' `lspci | grep NVIDIA`

# Prepare the dataset
unzip SMAP_Climate_In_Situ_Kenaston_training_data.csv

# Following the example from http://chtc.cs.wisc.edu/conda-installation.shtml
# except here we download the installer instead of transferring it
# Download a specific version of Miniconda instead of latest to improve
# reproducibility
export HOME=$PWD
wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
export PATH=$HOME/miniconda3/bin:$PATH

# Update conda as workaround for https://github.com/conda/conda/issues/9681
# Will no longer be needed once conda >= 4.8.3 is available from repo.anaconda.com
conda install conda=4.8.3

# Set up conda
source $HOME/miniconda3/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes --set changeps1 no

# Install packages specified in the environment file
conda env create -f environment.yml

# Activate the environment and log all packages that were installed
conda activate pytorch-gpu
conda list

# Modify these lines to run your desired Python script
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
python train.py --epochs=100 --early_stopping_patience=10 --direct_connection_from_previous_output=True --time_varying_features_name='prcp,srad,tmax,tmin,vp' --save_model='model_3.pt' --save_entire_model='model_entire_3.pt' --save_figure='model_3.png'