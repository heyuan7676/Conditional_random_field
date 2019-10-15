#!/bin/bash -l

#SBATCH --time 2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

BASEDIR=/scratch1/battle-fs1/heyuan/project_informative_looping
FROM_BEGINNING=false
TISSUE=Adipose_Subcutaneous

PATH=/scratch1/battle-fs1/heyuan/anaconda/bin:$PATH  ## python
PATH=/home/yhe23/R-3.2.0/bin:$PATH    ## R
PATH=$PATH:/scratch1/battle-fs1/tools_shared/bowtie-1.1.2 # bowtie
PATH=$PATH:/scratch1/battle-fs1/tools_shared/samtools-1.2  # samtools
PATH=$PATH:/scratch0/battle-fs1/heyuan/HOMER/./bin/  # homer

export PATH


LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/battle-fs1/tools_shared/gcc-4.9.2/lib:/scratch1/battle-fs1/tools_shared/gcc-4.9.2/lib64
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/battle-fs1/heyuan/anaconda/lib
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yhe23/R-3.2.0/lib
export LD_LIBRARY_PATH


mkdir ${BASEDIR}
mkdir ${BASEDIR}/plots
mkdir ${BASEDIR}/data
mkdir ${BASEDIR}/features


cd ${SLURM_SUBMIT_DIR}

################################################################### 
### 1. download and process data from ENCODE 
###################################################################

if ${FROM_BEGINNING}
then 
    bash seqdata_pipeline_fa.sh ${BASEDIR}  ${TISSUE}   ## ATAC-seq
    bash seqdata_pipeline_bam.sh ${BASEDIR} ${TISSUE}   ## CTCF ChIP-seq
    cp /scratch1/battle-fs1/heyuan/project_informative_looping/features/GSE63525_GM12878* ${BASEDIR}/features
    cp /scratch1/battle-fs1/heyuan/project_informative_looping/features/segments_Adipose_Subcutaneous.bed ${BASEDIR}/features
else
    echo "Not run"
    cp /scratch1/battle-fs1/heyuan/project_informative_looping/features/* ${BASEDIR}/features
fi



################################################################### 
### 2. annotate the matrixeQTL results with features and
###    extract the positive and negative sets
###################################################################

# bash makingSets.sh ${BASEDIR} ${TISSUE}



################################################################### 
### 3. run exploratory analysis and machine learning model
###################################################################

export TISSUE
/scratch1/battle-fs1/heyuan/anaconda/bin/python train_model.py ${BASEDIR} 






