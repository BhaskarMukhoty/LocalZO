# LocalZO
## Direct Training of SNN using Local Zeroth Order Method
Bhaskar Mukhoty,  Velibor Bojkovic, William de Vazelhes, Xiaohan Zhao, Giulia De Masi, Huan Xiong, Bin Gu

37th Conference on Neural Information Processing Systems

The code re-uses the following work:  
@inproceedings{deng2021temporal,
  title={Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting},
  author={Deng, Shikuang and Li, Yuhang and Zhang, Shanghang and Gu, Shi},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

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
Bhaskar Mukhoty (bhaskar.mukhoty@gmail.com)
