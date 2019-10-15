#!/bin/bash 

BASEDIR=$1
tissue=$2

DATADIR=/scratch1/battle-fs1/heyuan/project_informative_looping/annotations
FEATUREDIR=${BASEDIR}/features

BT_DIR=/scratch1/battle-fs1/tools_shared/bowtie2-2.2.8
SAMTOOLS_DIR=/scratch1/battle-fs1/tools_shared/samtools-1.2
LOGFILE=logfile.txt


[ -z "$DATADIR" ] && { echo "Need to set BASEDIR"; exit 1; }
[ -z "$BT_DIR" ] && { echo "Need to set Bowtie Dir"; exit 1;  }
[ -z "$SAMTOOLS_DIR" ] && { echo "Need to set SAMTOOLS_DIR"; exit 1;  }

echo "Processing CTCF ChIP-seq data"

########################################### 
### 1. sam to bam conversion 
###     - filter out low quality reads;
###     - sort the bam file by read names (for conversion to bed files)
########################################### 

cd ${SLURM_SUBMIT_DIR}

echo "sam to bam files conversion"


##### ChIP-seq reads (from bam files)

${SAMTOOLS_DIR}/samtools view -h ${DATADIR}/ChIP_seq/ENCFF997ZMP.bam | ./filtersam.awk | ${SAMTOOLS_DIR}/samtools sort -n -@ 3 - ${DATADIR}/ChIP_seq/${tissue}_sorted


##### bam to bed conversion

# ${SAMTOOLS_DIR}/samtools index ${DATADIR}/${tissue}_sorted.bam
# bedtools bamtobed –i ${tissue}_sorted.bam



########################################### 
### 2. peak calling using homer and macs2
########################################### 


## ChIP-seq

DIR=${DATADIR}/ChIP_seq
mkdir ${DIR}/homer
mkdir ${DIR}/macs



echo "Start peak calling using homer"
start=`date +%s`

makeTagDirectory ${DIR}/homer ${DIR}/${tissue}_sorted.bam
findPeaks ${DIR}/homer -o ${DIR}/${tissue}_CTCF_homer.txt -localSize 50000 -size 150 -minDist 50
cat ${DIR}/${tissue}_CTCF_homer.txt | ./filterhomer.awk > ${FEATUREDIR}/${tissue}_CTCF_homer.bed

end=`date +%s`
time=$((end-start))
echo "Finished in "${time}" s"




echo "Start peak calling using macs2"
start=`date +%s`

macs2 callpeak -t ${DIR}/${tissue}_sorted.bam -f BAM -g hs -n ${tissue}_CTCF_macs --outdir ${DIR}/macs
mv ${DIR}/macs/${tissue}_CTCF_macs_summits.bed ${FEATUREDIR}

end=`date +%s`
time=$((end-start))
echo "Finished in "${time}" s"










 
