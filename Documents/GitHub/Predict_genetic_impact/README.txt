To run the code:
sbatch run.sh





Code composition:

   run.sh: runs the whole process. You can change the base directory in run.sh. ( Please note that no / at the end)

      - seqdata_pipeline_fa.sh: process fastq files for ATAC-seq data. 
                                  Alignment using bowtie2 takes 2.73 hours, file consversion takes 16 min, 
                                peak calling takes 22 min using homer, and 17 min using macs2.
                                  This was largely because of the deep sequencing in this dataset, where 104
                                million paired reads were obtained.
 
      - seqdata_pipeline_bam.sh: process bam files for ChIP-seq data.
                                  Running it takes 10 min in total (which resulted in very sparse signal)

        * seqdata_pipeline_fa.sh and seqdata_pipeline_bam.sh were not run to avoid too long time of running. 
          Processed data will be copied to the directory.
 
        * You could change FROM_BEGINNING=true to run these two commands

      - makingSets.sh: annotate the matrixeQTL results with the features; extract the positive and negative sets

      - train_model.py: run exploratory analysis and machine learning model

          - features.txt: features to put in the model 

          - chrlist.txt: chromosomes that provide the data
