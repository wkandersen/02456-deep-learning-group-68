#!/bin/bash
#BSUB -j sweep_fashion_deep-learning
#BSUB -o bash_outputs/sweep_fashion_%J.out
#BSUB -e bash_outputs/sweep_fashion_%J.err
#BSUB -R "rusage[mem=2G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 23:00
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpuv100
#BSUB -B
#BSUB -n 4

source ~/miniconda3/bin/activate
conda activate computer_vison
python src/sweep_train_fashion_mnist.py