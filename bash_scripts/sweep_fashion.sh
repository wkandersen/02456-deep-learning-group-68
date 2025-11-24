#!/bin/bash
#BSUB -J sweep_fashion_deep-learning
#BSUB -o bash_outputs/sweep_fashion_%J.out
#BSUB -e bash_outputs/sweep_fashion_%J.err
#BSUB -R "rusage[mem=3G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -q hpc
#BSUB -B
#BSUB -n 4



source ~/miniconda3/bin/activate
conda activate computer_vison

python src/sweep_train_fashion10.py
