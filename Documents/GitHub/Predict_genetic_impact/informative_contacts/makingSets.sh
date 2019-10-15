#!/bin/bash 

BASEDIR=$1
TISSUE=$2

#################################################################
## shrink the dataset, to reduce time reading in
#################################################################

# tissue=Adipose_Subcutaneous
# cd /scratch0/battle-fs1/heyuan/long_range/SNP_gene/${tissue}/


# for fn in *filtered.txt
# do
#     echo $fn
#     head -n1 ${fn} > temp
#     awk '{if($6<0.1) print $0}' ${fn} >> temp
#     awk '{if($6>0.9995) print $0}' ${fn} >> temp
#     mv temp ${fn%.txt}_subset.txt
# done
    



#################################################################
## making the positive sets and negative sets
#################################################################

for i in {22..21}
do
    echo chr${i}
    /home/yhe23/R-3.2.0/bin/Rscript makingSets.R ${TISSUE} chr${i} ${BASEDIR}
done
