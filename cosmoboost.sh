#!/bin/bash
#SBATCH --job-name=cosmoboost0
#SBATCH --output=output.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=50GB
#SBATCH --mail-user=maamari@usc.edu

cd /home1/maamari/maamari/test/CosmoBoost
python3 odeTest.py
