#!/bin/bash
## sbatch job.sh to run
#SBATCH --job-name=EA2
#SBATCH --mail-user=jiuqian@umich.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=stats_dept1
#SBATCH --partition=standard

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

## 5GB/cpu is the basic share
#SBATCH --mem-per-cpu=5GB

## wall time hours:minutes:seconds
#SBATCH --time=5:00:00
#SBATCH --array=0,10,20,30,40,50,60,70,80,90

###   Load software modules
####  Commands your job should run follow this line
###   Load software modules
module load python3.11-anaconda/2024.02
#### source $(conda info --base)/etc/profile.d/conda.sh
#### conda activate flost-env

####  Commands your job should run follow this line
/home/jiuqian/.conda/envs/flost-env/bin/python job_simu.py $SLURM_ARRAY_TASK_ID
#### python3 job_simu.py $SLURM_ARRAY_TASK_ID 
