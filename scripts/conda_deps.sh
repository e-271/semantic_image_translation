set -ex
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing ipdb scikit-image
conda install pytorch=0.4.1 torchvision cudatoolkit=9.0 -c pytorch # add cuda90 if CUDA 9
conda install visdom dominate -c conda-forge # install visdom and dominate
