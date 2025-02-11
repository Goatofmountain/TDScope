
''''
__Author__: Kailing Tu
__Version__: v18.0.0
__ReleaseTime__: 2024-11-26
Requirement:
    pysam v0.19.1
    spoa
    time
    numpy v1.21.5
    pandas v1.3.4
    mappy v2.20
Description: Parsing ONT bam file to find aim read sequence and filter out low quality reads for further analysis
    Modular components:
        MethodTestPipe
            Function to judge whether target TD have somatic event and return read label dataframe, usually used as Function test for author TKL
        Decision:
            Function to judge whether target TD have somatic event and return formal bed output with format:
            [
            'chrom', 'start', 'end',                                                            # following reference genome location
            'consensus somatic sequences', 'somatic support readIDs', 'Number of somTD Type',   # somatic information 
            'consensus germline sequences', 'germline support readIDs', 'Number of germTD Type' # germline information 
            'Pvalue'                                                                            # additional pvalue for inter clusters test 
            ]
'''

from ReadsCluster import *
from DataScanner import *
import numpy as np
import pandas as pd
from spoa import poa
from scipy.stats import chi2_contingency, chi2
import time
from Levenshtein import distance as levenshtein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def TestSom(Control, Case):
    # Input Control 2d array and Case 2d array 
    # H0: Control and Case share same multinormal distribution parameter 
    # H1: Control and Case have different multinormal distribution parameter 
    Control_freq = np.eye(5)[Control.astype(int)].sum(axis=0)
    Case_freq = np.eye(5)[Case.astype(int)].sum(axis=0)
    testTable = np.array([chi2_contingency([Case_freq[i]+1, Control_freq[i]+1], lambda_="log-likelihood") for i in range(Case_freq.shape[0])], dtype=object)
    pvalue = 1 - chi2.cdf(testTable[:, 0].sum(), df=testTable.shape[0])
    return(pvalue)

def MethodTestPipe(refFile, bamFileList, LabelList, TDRecord,Tlabel='tumor', readcutoff=3):
    # For Pipeline Test 
    seqencode_New, seqdatamx, ReadIDs = DataMaker(refFile, bamFileList, LabelList, TDRecord)
    ReadLabels1 = np.array([x.split("|")[0].split("_")[-1] for x in ReadIDs])
    K, seqdatamx, DatLabel, thetap, gamma, pie, BICList = EMCluster(seqdatamx, initselection=1)
    DistDf = pd.DataFrame.from_records(seqencode_New)
    DistDf.index = ReadIDs
    DistDf['label1'] = ReadLabels1
    DistDf['clusterID'] = DatLabel
    DistDf.sort_values(['clusterID'], inplace=True)
    # Check For each cluster and find potential somatic event 
    ClusterAnno = {}
    for L in np.unique(DatLabel):
        ReadIDsub = np.array(ReadIDs)[np.where(DatLabel==L)[0]]
        ReadType = np.unique([x.split("|")[0].split("_")[-1] for x in ReadIDsub])
        sampleList = list(np.unique([x.split("|")[0] for x in ReadIDsub]))
        if (ReadType.shape[0] == 1) and (ReadType[0]==Tlabel) and (ReadIDsub.shape[0]>=readcutoff):
            print('%s find somatic event at %s' % (",".join(sampleList), TDRecord))
            ClusterAnno[L] = 'somatic'
        else:
            ClusterAnno[L] = 'germline'
    DistDf['clusterAnno'] = DistDf['clusterID'].map(ClusterAnno)
    return(DistDf)

def FindSomClust(sequences, readIDList, Tlabel='tumor'):
    # test if cluster contain somatic event 
    n = len(sequences)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):  # 距离矩阵是对称的
            dist = levenshtein_distance(sequences[i], sequences[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    features = mds.fit_transform(distance_matrix)
    bic_scores = []
    n_clusters_options = range(1, np.min([n,11]))  # test from 1 to 10 
    modelList = []
    for k in n_clusters_options:
        gmm = GaussianMixture(n_components=k, random_state=42).fit(features)
        modelList.append(gmm)
        bic_scores.append(gmm.bic(features))
    bestKIDX = np.argmin(bic_scores)
    n_k = n_clusters_options[bestKIDX]
    model = modelList[bestKIDX]
    labels = model.predict(features)
    # Test somatic 
    SomClust = []
    GermClust = []
    Tags = np.array([x.split("|")[0].split("_")[-1] for x in readIDList])
    for L in np.unique(labels):
        ClustTags = Tags[np.where(labels==L)[0]]
        if len(ClustTags) == len([x for x in ClustTags if x==Tlabel]):
            SomClust.append(L)
        else:
            GermClust.append(L)
    return(np.array(SomClust), np.array(GermClust), labels)

def Decision(TDRecord, sequenceList, ReadIDs, flank_5,flank_3, windowFlag='NormalOutput', Tlabel='tumor', readcutoff=3, hcutoff=3, scutoff=0.05):
    '''
    Decision for somatic SV
    :param:
        TDRecord: bed record for candidate region, should contain at least 3 columns in tab split, chrom\tstart\tend
        sequenceList: subsequence list with reference sequence on the top  
        ReadIDs: readIDList
        flank_5: 5' flank sequence in ATCG
        flank_3: 3' flank sequence in ATCG
        Tlabel: keyword representing for tumor, by default "tumor"
        readcutoff: somatic support reads cutoff, by default 3
        offset: flank sequence length, by default 200
        hcutoff:
    '''
    # Pipeline for formal run 
    chrom,start,end = TDRecord.strip().split("\t")[0:3]
    ReadTags, Tagcount = np.unique(np.array([x.split("|")[0].split("_")[-1] for x in ReadIDs]), return_counts=True)
    record = [chrom, start, end,
              "-",
              "-",
              0,
              "-",
              "-",
              0,windowFlag]
    if (len(sequenceList) > 3) and (ReadTags.shape[0]>=2) and (np.min(Tagcount)>=3):
        # Make a simple decision
        sequences = sequenceList[1:]
        SomClust, GermClust, labels = FindSomClust(sequences, ReadIDs, Tlabel)
        if len(SomClust) > 0:
            record[4] = ";".join([",".join(list(ReadIDs[np.where(labels==x)[0]])) for x in np.unique(SomClust)])
            record[7] = ";".join([",".join(list(ReadIDs[np.where(labels==x)[0]])) for x in np.unique(GermClust)])
            record[-1] = windowFlag+"|FastSelected"
            # Tandem Repeat Regions with somatic supported reads more than 3 should be further analysis with sequence specific model
            if len([x for x in labels if x in SomClust]) >= readcutoff:
                seqencode_New, seqdatamx, ReadIDs = MSAFeatureSelection(sequenceList, flank_5, flank_3, ReadIDs, hcutoff=hcutoff, scutoff=scutoff)
                if (seqdatamx.shape[0] != 0) and (seqdatamx.shape[1]>=10):
                    K, seqdatamx, DatLabel, thetap, gamma, pie, BICList = EMCluster(seqdatamx, initselection=1)
                    # Check For each cluster and find potential somatic event 
                    ClusterAnno = {}
                    somaticReadIDXCollect = []
                    somaticSeqCollect = []
                    germlineReadIDXCollect = []
                    germlineSeqCollect = []
                    for L in np.unique(DatLabel):
                        ReadIDsub = np.array(ReadIDs)[np.where(DatLabel==L)[0]]
                        ReadType = np.unique([x.split("|")[0].split("_")[-1] for x in ReadIDsub])
                        if (ReadType.shape[0] == 1) and (ReadType[0]==Tlabel) and (ReadIDsub.shape[0]>=readcutoff):
                            ClusterAnno[L] = 'somatic'
                            somaticReadIDXCollect.append(np.where(DatLabel==L)[0])
                        else:
                            ClusterAnno[L] = 'germline'
                            if np.where(DatLabel==L)[0].shape[0] >= readcutoff:
                                germlineReadIDXCollect.append(np.where(DatLabel==L)[0])
                    if len(somaticReadIDXCollect) > 0:     # at least one Somatic SV exist 
                        for somIDX in somaticReadIDXCollect:
                            somSequence = list(map(SeqDecoder, seqencode_New[somIDX+1]))
                            SeqLen = np.max([len(x) for x in somSequence])
                            if SeqLen > 0:
                                consensus, msa = poa(somSequence,1) 
                                somConsSeq = consensus
                                somaticSeqCollect.append(somConsSeq)
                            else:
                                somConsSeq = "-"
                                somaticSeqCollect.append(somConsSeq)
                    if len(germlineReadIDXCollect) > 0:
                        for germIDX in germlineReadIDXCollect:
                            germSequence = list(map(SeqDecoder, seqencode_New[germIDX+1]))
                            SeqLen = np.max([len(x) for x in germSequence])
                            if SeqLen > 0:
                                consensus, msa = poa(germSequence,1) 
                                germConsSeq = consensus
                                germlineSeqCollect.append(germConsSeq)
                            else:
                                germConsSeq = '-'
                                germlineSeqCollect.append(germConsSeq)
                    # Write Out Record 
                    if (len(somaticSeqCollect) > 0) and (len(germlineReadIDXCollect) > 0):
                        # Collect somatic pvalue 
                        somPCollect = []
                        # for somIDX in somaticReadIDXCollect:
                        # somPCollect.append(np.max([TestSom(seqdatamx[germIDX], seqdatamx[somIDX]) for germIDX in germlineReadIDXCollect]))
                        record = [chrom, start, end,
                                ";".join(somaticSeqCollect),
                                ";".join([",".join(list(np.array(ReadIDs)[somIDX])) for somIDX in somaticReadIDXCollect]),
                                len(somaticSeqCollect),
                                ";".join(germlineSeqCollect),
                                ";".join([",".join(list(np.array(ReadIDs)[germIDX])) for germIDX in germlineReadIDXCollect]),
                                len(germlineSeqCollect),
                                windowFlag+"|EMOutput"]
    return(record)








