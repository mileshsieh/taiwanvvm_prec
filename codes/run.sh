#!/usr/bin/bash
#SBATCH -J conv_data    # Job name
#SBATCH -p all          # job partition
#SBATCH -N 1            # Run all processes on a single node	
#SBATCH -c 1            # cores per MPI rank
#SBATCH -n 32           # Run a single task
#SBATCH -o conv_data.out  # output file
#SBATCH -e conv_data.err  # error file
#SBATCH --nodelist=mogamd

python save_wind_preci_data.py
#python conv_dataset.py

