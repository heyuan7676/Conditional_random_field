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


echo "Process ATAC-seq data"


########################################### 
### 0. Download ATAC-seq data 
###########################################

# cd ${DATADIR}/ATAC_seq

# wget -O rep1.fastq.gz https://www.encodeproject.org/files/ENCFF987RZW/@@download/ENCFF987RZW.fastq.gz
# echo "rep1"
# wget -O rep2.fastq.gz https://www.encodeproject.org/files/ENCFF215GFE/@@download/ENCFF215GFE.fastq.gz
# echo "rep2"
# gunzip rep1.fastq.gz
# gunzip rep2.fastq.gz




########################################### 
### 1. alignment using bowtie   
########################################### 

## download GRCh38 index
# cd ${DATADIR}/reference
# wget ftp://ftp.ncbi.nlm.nih.gov/genomes/archive/old_genbank/Eukaryotes/vertebrates_mammals/Homo_sapiens/GRCh38/seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.bowtie_index.tar.gz

echo "Align fastq files using bowtie2"


start=`date +%s`

READ_LENGTH=$((`head -n2 ${DATADIR}/ATAC_seq/${tissue}_rep1.fastq | tail -n1 | wc -c`-1))
echo ${READ_LENGTH} >> ${LOGFILE}
cd ${BT_DIR}
./bowtie2\
     -q \
     --phred33 \
     -L 25 \
     -p 6 \
     -x ${DATADIR}/ATAC_seq/reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.bowtie_index \
     -1 ${DATADIR}/ATAC_seq/${tissue}_rep1.fastq \
     -2 ${DATADIR}/ATAC_seq/${tissue}_rep2.fastq \
     -S ${DATADIR}/ATAC_seq/${tissue}.sam

end=`date +%s`
time=$((end-start))
echo "Finished in "${time}" s"
echo 



########################################### 
### 2. sam to bam conversion 
###     - filter out low quality reads;
###     - sort the bam file by read names (for conversion to bed files)
########################################### 

cd ${SLURM_SUBMIT_DIR}

echo "sam to bam files conversion"

start=`date +%s`

cat ${DATADIR}/ATAC_seq/${tissue}.sam | \
    ./filtersam.awk | \
    ${SAMTOOLS_DIR}/samtools view -u - | \
    ${SAMTOOLS_DIR}/samtools sort -@ 10 - ${DATADIR}/ATAC_seq/${tissue}_sorted

end=`date +%s`
runtime=$((end-start))
echo "Finished in "${runtime}" s"
echo 




########################################### 
### 3. peak calling using homer and macs2
########################################### 

DIR=${DATADIR}/ATAC_seq
mkdir ${DIR}/homer
mkdir ${DIR}/macs

## homer

echo "Start peak calling using homer"
start=`date +%s`

makeTagDirectory ${DIR}/homer ${DIR}/${tissue}_sorted.bam
findPeaks ${DIR}/homer -o ${DIR}/${tissue}_ATAC_homer.txt -fdr 0.05
cat ${DIR}/${tissue}_ATAC_homer.txt | ./filterhomer.awk > ${FEATUREDIR}/${tissue}_ATAC_homer.bed

end=`date +%s`
time=$((end-start))
echo "Finished in "${time}" s"
echo

## macs2

echo "Start peak calling using macs"
start=`date +%s`

macs2 callpeak -t ${DIR}/${tissue}_sorted.bam -f BAM -g hs -n ${tissue}_ATAC_macs --nomodel --shift -100 --extsize 200 --outdir ${DIR}/macs
mv ${DIR}/macs/${tissue}_ATAC_macs_summits.bed ${FEATUREDIR}

end=`date +%s`
time=$((end-start))
echo "Finished in "${time}" s"
echo


mv ${DATADIR}/segments/segments_${tissue}.bed ${FEATUREDIR}







 
