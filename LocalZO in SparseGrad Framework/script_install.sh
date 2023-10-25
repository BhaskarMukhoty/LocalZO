source /apps/local/conda3.1-init.sh
conda env create -f environment.yml
conda activate s3gd;
cd cuda
python setup_s3gd_cuda.py install