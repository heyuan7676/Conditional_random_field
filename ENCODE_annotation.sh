


### Download site: https://www.encodeproject.org/matrix/?type=Experiment&assay_title=ATAC-seq&assay_title=DNase-seq&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_type=tissue&organ_slims=lung&organ_slims%21=kidney&organ_slims=brain&organ_slims=stomach&organ_slims=small+intestine&organ_slims=adrenal+gland&organ_slims%21=extraembryonic+component&organ_slims%21=thymus&organ_slims=pancreas&organ_slims%21=placenta&organ_slims%21=gonad&organ_slims%21=spinal+cord&organ_slims%21=thyroid+gland&organ_slims=ovary&organ_slims%21=eye&organ_slims%21=epithelium&organ_slims=testis&organ_slims=breast&organ_slims=esophagus&organ_slims=adipose+tissue&organ_slims=artery&organ_slims=liver&organ_slims=spleen&organ_slims=uterus&organ_slims=skin+of+body&organ_slims=intestine&organ_slims=heart&organ_slims%21=large+intestine&organ_slims%21=ureter&organ_slims%21=musculature+of+body&organ_slims%21=limb&organ_slims%21=blood+vessel&organ_slims%21=embryo&organ_slims=prostate+gland&organ_slims=nerve&organ_slims%21=urinary+bladder&organ_slims%21=vasculature&organ_slims%21=connective+tissue&organ_slims%21=lymphoid+tissue&organ_slims%21=mouth&organ_slims%21=tongue

datadir="/scratch1/battle-fs1/heyuan/tissue_spec_eQTL/data/ENCODE"
tooldir="/scratch1/battle-fs1/heyuan/tissue_spec_eQTL/tools"


### download DNAse/ATAC for relevant tissues from ENCODE

cd ${datadir}
xargs -n 1 curl -O -L < ENCODE_download_used.txt

cp $bed.gz save/

for f in *bed.gz
do
	gunzip $f
	mv ${f%.gz} ${f%.bed.gz}.narrowbed
done


### convert narrowPeaks to bed files

for f in *bed
do
	newf=${f%.narrowbed}.bed
	paste <(cut -f 1-4 ${f}) <(cut -f 7 ${f}) <(cut -f 6 ${f}) > ${newf}
done

mv *.narrowbed narrowBed/


### extract genome information; tissue information
### lift GRCh38 to hg19


filename=`ls ${datadir}| grep "bed"`

cd ${tooldir}
for f in $filename
do
	expr=${f%.bed}
	genome=`cat ${datadir}/metadata.tsv | grep ^${expr} | cut -d'       ' -f 43`
	tissue=`cat ${datadir}/metadata.tsv | grep ^${expr} | cut -d'       ' -f 7`
	tissue_condensed=${tissue// /_}
	newfn=${f%.bed}_${tissue_condensed}_${genome}.bed
	mv ${datadir}/$f ${datadir}/${newfn}
	if [[ $genome == "GRCh38" ]]
	then
		echo $newfn
		./liftOver ${datadir}/${newfn} hg38ToHg19.over.chain.gz ${datadir}/${newfn//GRCh38/hg19} ${datadir}/${newfn%.bed}_unlifted.bed
	fi
done


rm *GRCh38*



### merge data for the same tissue

tissues=(stomach heart small_intestine adrenal_gland brain lung heart_left_ventricle frontal_cortex pancreas prostate_gland spleen uterus caudate_nucleus cerebellar_cortex cerebellum putamen skin_of_body muscle tibial_artery tibial_nerve)
for t in ${tissues[@]}
do
	echo $t
	echo *${t}* > ${t}_included_files.txt
	cat *${t}* > ${t}.bed
	cat ${t}.bed | sort -k1,1 -k2,2n | bedtools merge -c 5 -o mean > ${t}_merged.bed
done











