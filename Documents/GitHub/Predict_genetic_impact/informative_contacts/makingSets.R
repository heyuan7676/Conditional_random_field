#!/usr/bin/Rscript

suppressMessages(library(data.table))
suppressMessages(library(GenomicRanges))
suppressMessages(library(dplyr))


args <- commandArgs(TRUE)
tissue = args[1]   # Adipose_Subcutaneous
chr = args[2]
basedir = args[3]

input_dir = '/scratch0/battle-fs1/heyuan/long_range/SNP_gene/'
feature_dir = paste0(basedir, "/features/")
output_dir = paste0(basedir, "/data/")


###################################################################################
#### generateSets(): extract positive and negatvie sets from matrixeQTL results,
####                 all pairs are expected to have hic contacts > threshold
#### hic_threshold: the threshold to define whether the pair has contact
#### distance_to_match: the distance to match for negative pairs
###################################################################################

generateSets <- function(tissue, chr, hic_threshold = 0, distance_to_match = 5e3, QTL_FDR_threshold = 0.1){
    #### extract positive and negatvie sets from matrixeQTL results
    #### all pairs are expected to have hic contacts 
    print(chr)
    filename = paste0(input_dir,tissue,'/',tissue,'_',chr,'_windowsize_10000_trans_100kB_withhicdis_filtered_subset.txt')
    pairs = fread(filename, sep='\t',header=T,stringsAsFactors=F)
    pairs = as.data.frame(pairs)
    pairs = pairs[pairs$hicvalues > hic_threshold, ]    ## all pairs have hic values > hic_threshold
    ## extract positive set and negative pool
    pairs$FDR = as.numeric(pairs$FDR)
    QTLs = pairs[pairs$FDR<QTL_FDR_threshold,]
    #### restrict to at most 10 SNPs per gene
    set.seed(0)
    QTLs = do.call("rbind",as.list(by(QTLs,QTLs$gene, function(x) if(dim(x)[1]>10) {x[sample(nrow(x),10),]} else {x})))
    ### can pick 1 out of 1000 negative pairs to match for distance
    ### restrict the number of negative pairs to speed up
    # neg_FDR_threshold = quantile(pairs$FDR, 1-dim(QTLs)[1]*1000/dim(pairs)[1]) 
    neg_FDR_threshold = 0.99
    nega_sets = pairs[pairs$FDR > neg_FDR_threshold,]                            
    ### match for distance
    distance = unname(apply(QTLs[,c("start","end","pos")],1,function(x) min(abs(x[1]-x[3]),abs(x[2]-x[3])))) 
    nega_distance = unname(apply(nega_sets[,c("start","end","pos")],1,function(x) min(abs(x[1]-x[3]),abs(x[2]-x[3]))))
    set.seed(0)
    idx = sapply(distance,function(x) sample(c(-1,which(abs(nega_distance-x) < distance_to_match)),size=1))
    idx = idx[idx!=-1]
    random_nega_sets = nega_sets[idx,]
    ## combine the two sets
    sets = rbind(QTLs, random_nega_sets)
    sets$QTL = c(rep("QTL",dim(QTLs)[1]), rep("neg",dim(random_nega_sets)[1]))
    print(paste0("There are ", dim(QTLs)[1]," QTLs and ",dim(random_nega_sets)[1], " negative pairs in ", chr))
    return(sets)
}



###################################################################################
#### addfeatures(): to find the overlap part of the SNPs/genes and the feature,
####                and add the feature values to the SNPs/genes' locations
#### set_granges: the GRanges object of SNPs/genes
#### f_granges: the GRange object of the feature
###################################################################################

addfeatures <- function(set_granges, f_granges, aggregate = FALSE, takemean=TRUE){
    f_vec = rep(0,length(set_granges))
    if(aggregate){
        score(f_granges) = width(f_granges) * score(f_granges)
    }
    hits = findOverlaps(set_granges,f_granges)
    hits = data.frame(hits@queryHits,hits@subjectHits)
    hits$score = score(f_granges[hits$hits.subjectHits])
    if(takemean){
        ## if there is multiple hits in the feature object, use the mean value
        hits = hits %>% group_by(hits.queryHits) %>% summarise(score=mean(score))
    }else{
        hits = hits %>% group_by(hits.queryHits) %>% summarise(score=score[1])
    }
    f_vec[hits$hits.queryHits] = hits$score
    return(f_vec)
}




###################################################################################
#### annotate_features(): to annotate the SNPs/genes with the feature,
#### set: the dataframe of SNP-gene pairs
#### ft_fn: the filename of the feature 
####        (note that the features's columns should be in specific order)
###################################################################################

annotate_features <- function(set, ft_fn, window_size = 5000,header=F, aggregate = FALSE, takemean=TRUE){
    ft = read.table(paste0(feature_dir,ft_fn),header=header,stringsAsFactors=F)
    colnames(ft) = paste0("V",seq(1,dim(ft)[2]))
    ft_granges = GRanges(seq=ft$V1,ranges=IRanges(start=ft$V2, end=ft$V3),score=ft$V5)
    set_snps = GRanges(seq=set$chr.x, ranges=IRanges(set$pos-window_size,set$pos+window_size))
    set_genes = GRanges(seq=set$chr.x, ranges=IRanges(set$start-2000,set$end))   ## annotate the promoter region or the gene body?  both
    return(list(addfeatures(set_snps,ft_granges,aggregate = aggregate, takemean=takemean),addfeatures(set_genes,ft_granges,aggregate = aggregate, takemean=takemean)))
}



annotate_features_category <- function(set, ft_fn, window_size = 5000,header=F){
    ft = read.table(paste0(feature_dir,ft_fn),header=header,stringsAsFactors=F)
    colnames(ft) = paste0("V",seq(1,dim(ft)[2]))
    ## assume that column 6 contains the categorical information
    bins = seq(1, length(unique(ft$V6)))
    names(bins) = unique(ft$V6)
    bins["NoCategory"]= dim(ft)[2]+1
    ## convert categories into number to use GRanges overlap function
    score = as.vector(bins[ft$V6])
    ft_granges = GRanges(seq=ft$V1,ranges=IRanges(start=ft$V2, end=ft$V3),score=score)
    ## overlap
    set_snps = GRanges(seq=set$chr.x, ranges=IRanges(set$pos-window_size,set$pos+window_size))
    set_genes = GRanges(seq=set$chr.x, ranges=IRanges(set$start,set$end))
    snps = round(addfeatures(set_snps,ft_granges))
    genes = round(addfeatures(set_genes,ft_granges))
    snps[snps==0] =  dim(ft)[2]+1
    genes[genes==0] =  dim(ft)[2]+1
    ## convert back to categories
    return(list(names(bins)[snps], names(bins)[genes]))
}



###################################################################################
#### peaks(): to annotate the SNPs/genes with the peaks from sequence
###################################################################################

peaks <- function(set, tissue, chr, signal = 'ATAC'){
    if(!paste0(signal,"_homer_snp") %in% colnames(set)){
        homer = annotate_features(set, paste0(tissue, "_",signal,"_homer.bed"))
        set[,paste0(signal,"_homer_snp")] = homer[[1]]
        set[,paste0(signal,"_homer_gene")] = homer[[2]]
    }
    if(!paste0(signal,"_macs_snp") %in% colnames(set)){
        macs = annotate_features(set, paste0(tissue,"_",signal,"_macs_summits.bed"))
        set[,paste0(signal,"_macs_snp")] = macs[[1]]
        set[,paste0(signal,"_macs_gene")] = macs[[2]]
    }
    return(set)
}



###################################################################################
#### segway(): to annotate the SNPs/genes with the segment values from segway
###################################################################################

segway <- function(set, tissue, chr, signal = 'segments'){
    if(!paste0(signal,"_snp") %in% colnames(set)){
        segments = annotate_features(set, paste0(signal, "_",tissue,".bed"),header=T,aggregate = TRUE)
        set[,paste0(signal,"_segway_snp")] = segments[[1]]
        set[,paste0(signal,"_segway_gene")] = segments[[2]]
    }
    if(!paste0(signal,"_snp") %in% colnames(set)){
        segments = annotate_features_category(set, paste0(signal, "_",tissue,".bed"),header=T)
        set[,paste0(signal,"_segway_snp_bin")] = segments[[1]]
        set[,paste0(signal,"_segway_gene_bin")] = segments[[2]]
    }
    return(set)
}



###################################################################################
#### compartment(): to annotate the SNPs/genes with the comparmentations
####                1: A
####                2: B
####                0: no compartment
###################################################################################

compartment <- function(set, chr){
    comparts = read.table(paste0(feature_dir,'GSE63525_GM12878_subcompartments.bed'),header=F,stringsAsFactors=F)
    comparts = comparts[comparts$V5!=0, ]
    comparts = comparts[comparts$V1 == chr, ]
    comparts$midpoint = (comparts$V2 + comparts$V3) / 2
    set$compartment_snp = sapply(set$pos, function(x) comparts[which.min(abs(x-comparts$midpoint)),"V5"])
    set$compartment_gene = sapply(set$start, function(x) comparts[which.min(abs(x-comparts$midpoint)),"V5"])
    return(set)
}


###################################################################################
#### TADdomain(): to annotate the SNPs/genes with nearest TAD domains
###################################################################################

TADdomain <- function(set, chr){
    TADs = read.table(paste0(feature_dir,'GSE63525_GM12878_domains.bed'),header=F,stringsAsFactors=F)
    TADs = TADs[TADs$V1 == chr, ]
    TADs$midpoint = (TADs$V2 + TADs$V3)/2
    set$SNP_domain = sapply(set$pos, function(x) TADs[which.min(abs(x-TADs$midpoint)),"V5"])
    set$gene_domain = sapply(set$start, function(x) TADs[which.min(abs(x-TADs$midpoint)),"V5"])
    return(set)
}



###################################################################################
#### main(): wrap up
###################################################################################

main <- function(tissue, chr, from_start=TRUE){
    if(from_start){
        sets = generateSets(tissue, chr, QTL_FDR_threshold=0.05)
    }else{
        sets = read.table(paste0(output_dir, tissue, "_", chr, "_sets.txt"), header=T,stringsAsFactors=F)
    }
    sets = peaks(sets, tissue, chr, 'ATAC')
    sets = peaks(sets, tissue, chr, 'CTCF')
    sets = compartment(sets, chr)
    sets = segway(sets, tissue, chr, 'segments')
    sets = TADdomain(sets, chr)
    write.table(sets, paste0(output_dir, tissue, "_", chr, "_sets.txt"), row.names=F)
}

options(warn=-1)
main(tissue, chr,from_start=TRUE)
