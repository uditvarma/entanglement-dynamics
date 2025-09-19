#!/bin/bash

#SBATCH --clusters=amarel
#SBATCH --partition=main
#SBATCH --requeue
#SBATCH --job-name=KiRo6c
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G             
#SBATCH --time=05:00:00
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err

module purge 

module use /projects/community/modulefiles/

module load openmpi/4.1.6

module load julia

module load anaconda

source activate kicked_rotor_julia

cd /scratch/$USER/kicked_rotor_julia/scaling_small

mpiexec -n $SLURM_NTASKS julia --project=/scratch/et438/kicked_rotor_julia/julia_pr Data_for_scaling_K_4.0.jl

sacct --units=G --format=MaxRSS,MaxDiskRead,MaxDiskWrite,Elapsed,NodeList -j $SLURM_JOBID

echo ""
echo -n "Fair-share score: ";sshare | grep $USER | grep general | awk '{print $NF}'
echo ""


