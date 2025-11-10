#!/bin/bash
#BSUB -j sweep_cifar_deep-learning
#BSUB -o bash_outputs/sweep_cifar_%J.out
#BSUB -e bash_outputs/sweep_cifar_%J.err
#BSUB -R "rusage[mem=3G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpua40
#BSUB -B
#BSUB -n 4


conda activate computer_vison

source ~/miniconda3/bin/activate

python src/sweep_train_cifar10.py
