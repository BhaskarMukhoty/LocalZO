# Direct Training of SNN using Local Zeroth Order Method

Code re-uses the following:
@article{perez2021sparse,
  title={Sparse Spiking Gradient Descent},
  author={Perez-Nieves, Nicolas and Goodman, Dan FM},
  journal={Advances in Neural Information Processing Systems},
  year={2021} }

## Requirements

Required:
* A CUDA capable GPU with [CUDA 11.0](https://developer.nvidia.com/cuda-11.0-download-archive) installed
* [pyTorch 1.7.1](https://pytorch.org/get-started/previous-versions/#v171)
* [GCC >=5](https://gcc.gnu.org/) or equivalent C++ compiler

Optional:
* [matplotlib](https://matplotlib.org/stable/users/installing.html) (for plotting results)
* [scipy](https://matplotlib.org/stable/users/installing.html) (for plotting results)
* [pytables](https://www.pytables.org/usersguide/installation.html) (for reading .h5 files for SHD and N-MNIST datasets)

The code is developed on pyTorch 1.7.1 and CUDA 11.0

## Detailed setup

We assumme you have a CUDA capable GPU with [CUDA installed](https://docs.nvidia.com/cuda/). 

Create a new [Anaconda](https://docs.anaconda.com/anaconda/install/) environment and install the dependencies

```setup
conda env create -f environment.yml
```

Activate the environment
```setup
conda activate s3gd
```

Then install the torch extension. 
```setup
cd cuda
python setup_s3gd_cuda.py install
```

This should install the extension. If everything went fine you should end up getting something similar to 
```
...

Installed /home/USERNAME/.conda/envs/s3gd/lib/python3.8/site-packages/s3gd_cuda-0.0.0-py3.8-linux-x86_64.egg
Processing dependencies for s3gd-cuda==0.0.0
Finished processing dependencies for s3gd-cuda==0.0.0
```
You can test that the installation was successful by running the following
```
python
>>> import torch
>>> import s3gd_cuda
```

## Running 
You can now run the code
```
cd ..
python main.py
```

This will run with the Fashion-MNIST dataset by default.

If you wish to run the Spiking Heidelberg Digits (SHD) dataset you can [download](https://compneuro.net/posts/2019-spiking-heidelberg-digits/) it.
Then save the `train_shd.h5` and `test_shd.h5` in `datasets/SHD`.

You can also [download](https://www.garrickorchard.com/datasets/n-mnist) the Neuromorphic-MNIST dataset and convert the raw spikes into .h5 format. 
After downloading `Train.zip` and `Test.zip`, unzip them and copy the directories to `datasets/nmnist`. 
Then run

```
python datasets/nmnist/convert_nmnist2h5.py
```

Now you can specify the dataset that you wish to run
```
python main.py --dataset fmnist
python main.py --dataset nmnist
python main.py --dataset SHD
```

By default the code runs "--surrogate normal" if you wish to change the surrogate, please un-comment the corresponding line in the function surr_grad_spike_kernel inside cuda/s3gd_cud_kernel.cu