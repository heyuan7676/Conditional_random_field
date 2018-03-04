#!/bin/bash
#SBATCH --time 100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue

module load python
python CRF_all_edges.py -D $1 -M $2 -E $3 -S $4 -P $5 -B $6 -I $7 -G $8 -e ${9} -a ${10} -A ${11}
