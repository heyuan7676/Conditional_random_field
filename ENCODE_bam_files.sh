### Download site: https://www.encodeproject.org/matrix/?type=Experiment&assay_title=ATAC-seq&assay_title=DNase-seq&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_type=tissue&organ_slims=lung&organ_slims%21=kidney&organ_slims=brain&organ_slims=stomach&organ_slims=small+intestine&organ_slims=adrenal+gland&organ_slims%21=extraembryonic+component&organ_slims%21=thymus&organ_slims=pancreas&organ_slims%21=placenta&organ_slims%21=gonad&organ_slims%21=spinal+cord&organ_slims%21=thyroid+gland&organ_slims=ovary&organ_slims%21=eye&organ_slims%21=epithelium&organ_slims=testis&organ_slims=breast&organ_slims=esophagus&organ_slims=adipose+tissue&organ_slims=artery&organ_slims=liver&organ_slims=spleen&organ_slims=uterus&organ_slims=skin+of+body&organ_slims=intestine&organ_slims=heart&organ_slims%21=large+intestine&organ_slims%21=ureter&organ_slims%21=blood+vessel&organ_slims%21=embryo&organ_slims=prostate+gland&organ_slims=nerve&organ_slims%21=urinary+bladder&organ_slims%21=vasculature&organ_slims%21=connective+tissue&organ_slims%21=lymphoid+tissue&organ_slims%21=mouth&organ_slims%21=tongue&organ_slims=limb&organ_slims=musculature+of+body



datadir="/scratch1/battle-fs1/heyuan/tissue_spec_eQTL/data/TF"
tooldir="/scratch1/battle-fs1/heyuan/tissue_spec_eQTL/tools"


### download DNAse/ATAC for relevant tissues from ENCODE
### 602 files in total

cd ${datadir}
xargs -n 1 curl -O -L < ENCODE_bam.txt


### extract genome information; tissue information; sample information

### filter out fetal / child samples
### filter out broadPeak file, use narrowPeak for more precise information
### filter out pseudoreplicated idr thresholded peaks

### convert narrowPeaks to bed files
### lift GRCh38 to hg19

### 107 files remain

filename=`ls ${datadir}| grep "bam"`


for f in $filename
do
	expr=${f%.bam}
	filetype=`cat ${datadir}/metadata.tsv | grep ^${expr} | cut -d'       ' -f 2`
	pesudorep=`cat ${datadir}/metadata.tsv | grep ^${expr} | cut -d'       ' -f 3`
	tissue=`cat ${datadir}/metadata.tsv | grep ^${expr} | cut -d'       ' -f 7`
	sample_age=`cat ${datadir}/metadata.tsv | grep ^${expr} | cut -d'       ' -f 9`
	genome=`cat ${datadir}/metadata.tsv | grep ^${expr} | cut -d'       ' -f 43`
	if [[ "$filetype" == "bam" ]] && [[ "$sample_age" == "adult" ]] && [[ "$pesudorep" != "pseudoreplicated idr thresholded peaks" ]]
	then
		tissue_condensed=${tissue// /_}
		newfn=${f%.bam}_${tissue_condensed}_${genome}_sorted.bam
		mv ${datadir}/${f} ${datadir}/${newfn}
		if [[ $genome == "GRCh38" ]]
		then	
			echo $newfn
			CrossMap.py bam ${tooldir}/hg38ToHg19.over.chain.gz ${datadir}/${newfn} ${datadir}/${newfn//GRCh38/hg19}
		fi
	else
		mv ${datadir}/$f ${datadir}/notused
	fi
done



mv *GRCh38* GRCh38_notused



### sort and merge data for the same tissue

filename=`ls ${datadir}| grep "bam"`
for f in 
do
	samtools sort ${datadir}/${f} ${datadir}/${f%.bam}_sorted.bam
done


tissues=(stomach heart small_intestine adrenal_gland brain lung heart_left_ventricle frontal_cortex pancreas prostate_gland spleen uterus caudate_nucleus cerebellar_cortex cerebellum putamen skin_of_body muscle tibial_artery tibial_nerve)
for t in ${tissues[@]}
do
	echo $t
	echo *${t}*bed > ${t}_included_files.txt
	samtools merge ${bamin}/bcells_sorted.bam ${bamin}/bcells_sorted_1.bam ${bamin}/bcells_sorted_2.bam
done











