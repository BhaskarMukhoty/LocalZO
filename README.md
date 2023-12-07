# LocalZO
## Direct Training of SNN using Local Zeroth Order Method
Bhaskar Mukhoty,  Velibor Bojkovic, William de Vazelhes, Xiaohan Zhao, Giulia De Masi, Huan Xiong, Bin Gu

In 37th Conference on Neural Information Processing Systems

Link to the [paper](https://openreview.net/pdf?id=eTF3VDH2b6)

The code re-uses the following repositories:  
[Temporal Efficient Training](https://github.com/brain-intelligence-lab/temporal_efficient_training), [Sparse Spiking Gradient Descent](https://github.com/npvoid/SparseSpikingBackprop) 


## Prerequisites
The Following Setup is tested and it is working:
 * Python>=3.5
 * Pytorch>=1.9.0
 * Cuda>=10.2

## Usage:
#### LocalZO+TET
python3 ./main_training_parallel.py --lr 0.001 --T 6 --lamb 0.05 --epochs 300 --batch_size 64 --TET 1  --cut 1 --seed 1000 --dataset cifar10 --resume 0

python3 ./main_test.py --T 6 --TET 1 --cut 1 --dataset cifar10 --batch_size 64  
 
#### LocalZO+tdBN
python3 ./main_training_parallel.py --lr 0.001 --T 6 --lamb 0.05 --epochs 300 --batch_size 64 --TET 0  --cut 1 --seed 1000 --dataset cifar10 --resume 0

python3 ./main_test.py --T 6 --TET 0 --cut 1 --dataset cifar10 --batch_size 64

## Contact:
Bhaskar Mukhoty (firstname.lastname@gmail.com)
