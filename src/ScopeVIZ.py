'''
Clearly visualization of TDscope
1, Reads Extraction, sort reads into vline plot, 
2, Reads Feature selection;
3, Reads Clustering 
4, Somatic event selection
'''
import os,re 
import pandas as pd 
import numpy as np
import pysam
import networkx as nx
## EM Clustering 
from DataScanner import *
from ReadsCluster import *
from DecisionMaker import *
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt 
import time 
from scipy.stats import chi2_contingency
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from spoa import poa         # spoa version 0.2.1 
import logging
import argparse
import functools
from multiprocessing import Pool
np.random.seed(2023)

#  ************Settings************ # 
# 5色组合
base_colors = np.array(['#FFB74D', '#64B5F6', '#81C784', '#F06292'])
baseName = np.array(['A','T','C','G'])
gap_color = '#BDBDBD'
category_colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#BCBD22', '#7F7F7F']
ClusterDict = {"ref":0, 'normal':1, 'tumor':2}

#  ************Settings************ # 

#  ************Functions************ # 

# Data Scanner 
def SeqEncoder(seqinput):
    alphabet = {'A':0, 'T': 1, 'C':2, 'G':3, '-':4}
    encodeList = []
    for s in seqinput:
        encodeList.append(alphabet[s.upper()])
    return(np.array(encodeList))

def SeqDecoder(seqinput):
    alphabet = {0:'A', 1:'T', 2:'C', 3:'G', 4:'-'}
    decodeseq = ''
    for s in seqinput:
        if s != 4:
            decodeseq += alphabet[s]
    return(decodeseq)

def SeqAligner(seqList):
    # Input selected sequence list 
    # return MSA result for these sequences
    consensus, msa = poa(seqList, 1)
    seqdatamx = np.array(list(map(SeqEncoder, msa)))
    return(seqdatamx)

def CallMargin_f5(msa, flank_5):
    # select TD start and end region for further analysis 
    ## check the hg38 reference for msa columns setting 
    examplesequence = msa[0]
    IDXPool = []
    tmpflank = ''
    for I in range(len(examplesequence)):
        if examplesequence[I] !="-":
            tmpflank += examplesequence[I]
            IDXPool.append(I)
        if tmpflank == flank_5:
            break
    return(IDXPool)

def CallMargin_f3(msa, flank_3):
    # select TD start and end region for further analysis 
    ## check the hg38 reference for msa columns setting 
    examplesequence = msa[0]
    IDXPool = []
    tmpflank = ''
    for I in range(len(examplesequence)-1, 0, -1):
        if examplesequence[I] !="-":
            tmpflank = examplesequence[I] + tmpflank
            IDXPool.append(I)
        if tmpflank == flank_3:
            break
    return(IDXPool)

def CallMargin(msa,flank_5,flank_3):
    # select TD start and end region for further analysis 
    ## check the hg38 reference for msa columns setting 
    examplesequence = msa[0]
    IDXPool = []
    tmpflank = ''
    for I in range(len(examplesequence)):
        if examplesequence[I] !="-":
            tmpflank += examplesequence[I]
            IDXPool.append(I)
        if tmpflank == flank_5:
            break
    tmpflank = ''
    for I in range(len(examplesequence)-1, 0, -1):
        if examplesequence[I] !="-":
            tmpflank = examplesequence[I] + tmpflank
            IDXPool.append(I)
        if tmpflank == flank_3:
            break
    return(np.array(IDXPool))

def FindNonSameSite(seqencode_New_Sub, cutoff=3):
    featureExists = np.where(np.isnan(seqencode_New_Sub), 0, 1)
    TotalCount = featureExists.sum(axis=0)
    FeatureCount = []
    for Label in range(5):
        tmpFeatureCount = np.zeros((seqencode_New_Sub.shape[1],))
        ColIDX,Count = np.unique(np.where(seqencode_New_Sub==Label)[1], return_counts=True)
        for I in range(ColIDX.shape[0]):
            tmpFeatureCount[ColIDX[I]] = Count[I]
        FeatureCount.append(tmpFeatureCount)
    FeatureCountArr = np.array(FeatureCount)
    NonSameSiteIDX = np.where(np.sort(FeatureCountArr, axis=0)[-2] >= cutoff)[0]
    return(NonSameSiteIDX)

# Data Scanner 
# Df function Loading # 

def CountSomTD(Record,CA_type):
    # 卡方统计，判别高频TD是否来源于特定类型的肿瘤
    TypeName,TotalNum = np.unique(CA_type, return_counts=True)
    Type_TD = np.zeros((TypeName.shape[0],))
    TDType,TDCount = np.unique(CA_type[np.where(Record!=0)[0]], return_counts=True)
    for T in TDType:
        Type_TD[np.where(TypeName == T)[0]] += TDCount[np.where(TDType==T)[0]]
    Type_TD_Non = TotalNum - Type_TD
    return(np.vstack([Type_TD, Type_TD_Non]))

def CountPatient(WindowPatientCount, TotalPatientCount):
    # 计算各癌种的正负分布
    PosCount = np.zeros(TotalPatientCount.shape[1])
    for i in range(WindowPatientCount.shape[1]):
        CAtype, Count = WindowPatientCount[:,i]
        PosCount[np.where(TotalPatientCount[0,:]==CAtype)[0]] += Count
    NegCount = TotalPatientCount[1,:] - PosCount
    return(np.vstack([PosCount, NegCount]))

def R_OE(crossTab):
    # crossTab is a 2D numpy array
    # First row: number of positive cases for each cancer type
    # Second row: number of negative cases for each cancer type
    # Extract observed positive cases
    Observed = crossTab[0, :]
    # Total number of positive and negative cases across all cancer types
    total_positive = crossTab[0, :].sum()
    total_negative = crossTab[1, :].sum()
    total_cases = total_positive + total_negative
    # Calculate expected positive cases for each cancer type
    cancer_type_totals = crossTab.sum(axis=0)
    Expected = (cancer_type_totals * total_positive) / total_cases
    # Calculate R(O/E) for each cancer type
    ROE = Observed / Expected
    return ROE

def AnnoR_OE(Record, CA_type):
    TotalPatientCount = np.vstack(np.unique(CA_type,return_counts=True))
    PositivePatient = Record.loc[(Record==1)].index
    WindowPatientCount = np.vstack(np.unique(CA_type.loc[PositivePatient], return_counts=True))
    crossTab = CountPatient(WindowPatientCount, TotalPatientCount)
    ROE = R_OE(crossTab)
    chi2, pvalue, dof, ex = chi2_contingency(crossTab)
    return(np.array(list(ROE)+[pvalue]), crossTab)

def AnnoTDwindow(Record, CA_type):
    AnnoSTR = '-'
    crossTab = CountSomTD(Record,CA_type)
    ROE = R_OE(crossTab)
    TypeName,TotalNum = np.unique(CA_type, return_counts=True)
    chi2, pvalue, dof, ex = chi2_contingency(crossTab)
    enrichedType = np.array([])
    if pvalue <= 0.05:
        # enrichedType = TypeName[np.where((ROE>=1.5)&(crossTab[0,:]>=5))[0]]
        enrichedType = TypeName[np.where((ROE>=1.5)&(crossTab[0,:]>=4))[0]]
    if enrichedType.shape[0] > 0:
        AnnoSTR = ",".join(list(enrichedType))
    return(AnnoSTR)

def FetchTDsubSeq(refFile, bamFileList, LabelList, TDRecord, offset=50):
    # Input refFile, bamFile, bamLabel, TDRecord 
    # get TD associated sub sequence from each reads 
    TDchrom, TDStart, TDEnd = TDRecord.strip().split("\t")[0],int(TDRecord.strip().split("\t")[1]), int(TDRecord.strip().split("\t")[2])
    F5start,F5end,F3start,F3end = TDStart-offset, TDStart, TDEnd,TDEnd+offset
    readIDList, readTDSeq, FlankMQ = [],[],[]   # readID, readSequence, primary ALN mapQ
    for bamIDX in range(len(bamFileList)):
        tmpReadSeqArr = []                          # readID => readsequence extracted from primary alignment, mapQ
        F5_readsIDX, F3_readsIDX = [], []
        F5_readName, F3_readName = [], []
        for reads in pysam.AlignmentFile(bamFileList[bamIDX]).fetch(TDchrom, TDStart, TDEnd):
            # primary alignment reads get the whole reads sequence 
            if not(reads.is_secondary or reads.is_supplementary):
                tmpReadSeqArr.append([reads.query_name, reads.query_sequence, reads.mapq])
            # Work For F5 flank 
            if (reads.reference_start<F5start) and (reads.reference_end>F5end) and (not reads.is_secondary):
                offset = 0
                if reads.is_supplementary:
                    CIGAR = np.array(reads.cigartuples)
                    if CIGAR[0][0] == 5:
                        offset = CIGAR[0][1]
                F5_readsIDX.append([reads.qname] +  ReadsLoci(reads, F5start, F5end, offset))
                F5_readName.append(reads.qname)
            # Work For F3 flank 
            if (reads.reference_start<F3start) and (reads.reference_end>F3end) and (not reads.is_secondary):
                offset = 0
                if reads.is_supplementary:
                    CIGAR = np.array(reads.cigartuples)
                    if CIGAR[0][0] == 5:
                        offset = CIGAR[0][1]
                F3_readsIDX.append([reads.qname] +  ReadsLoci(reads, F3start, F3end, offset))
                F3_readName.append(reads.qname)
        # remove reads 
        F5_N, F5_C = np.unique(F5_readName, return_counts=True)
        blackList_F5 = F5_N[np.where(F5_C>=2)[0]]
        F3_N, F3_C = np.unique(F3_readName, return_counts=True)
        blackList_F3 = F3_N[np.where(F3_C>=2)[0]]
        blackList = np.union1d(blackList_F3, blackList_F5)
        # Fetch 5Flank + TD + 3Flank sequence 
        if len(F5_readsIDX) * len(F3_readsIDX) * len(tmpReadSeqArr) > 0:
            spanReadIDs = np.intersect1d(np.intersect1d(np.array(tmpReadSeqArr)[:,0], np.array(F5_readsIDX)[:,0]), np.array(F3_readsIDX)[:,0])
            if blackList.shape[0] > 0:
                spanReadIDs = np.setdiff1d(spanReadIDs, blackList)
            if spanReadIDs.shape[0] >= 3:
                F5_df = pd.DataFrame(F5_readsIDX, columns=['readID', 'start', 'end'])
                F3_df = pd.DataFrame(F3_readsIDX, columns=['readID', 'start', 'end'])
                SeqDf = pd.DataFrame(tmpReadSeqArr, columns=['readID', 'qseq', 'mapQ'])
                SeqDf.index = SeqDf['readID']
                SummaryDf = pd.concat([F5_df.loc[F5_df['readID'].isin(spanReadIDs)].groupby(['readID'])['start'].apply(min), 
                                       F3_df.loc[F3_df['readID'].isin(spanReadIDs)].groupby(['readID'])['end'].apply(max), 
                                       SeqDf.loc[spanReadIDs, ['qseq', 'mapQ']]], axis=1)
                SummaryDf['SubSeq'] = SummaryDf.apply(lambda x: x['qseq'][x['start']:x['end']].replace("N",""), axis=1)
                readIDList += [LabelList[bamIDX] + "|" + x for x in SummaryDf.index]
                readTDSeq += [x for x in SummaryDf['SubSeq']]
                FlankMQ += [int(x) for x in SummaryDf['mapQ']]
    return(readTDSeq, readIDList, FlankMQ)

def SequencePlotRaw(ax, seqdatamx_Raw, readID, title='RawSeq',
                    base_colors=np.array(['#FFB74D', '#64B5F6', '#81C784', '#F06292']),
                    baseName = np.array(['A','T','C','G']), 
                    gap_color = '#999999', ClusterDict=ClusterDict, 
                    category_colors=category_colors):
    # Plot for raw extracted sequence 
    L = np.max([len(x) for x in seqdatamx_Raw])
    LabelX1,LabelX2, LabelYMIN, LabelYMAX, ClusterL, CAtypeL = [],[],[],[],[],[]
    PlotIdx = 0 
    for IDX in range(len(seqdatamx_Raw)):
        Dat = SeqEncoder(np.array(list(seqdatamx_Raw[IDX])))
        C = readID[IDX].split("|")[0]
        if np.where(Dat==4)[0].shape[0] > 0:
            ax.hlines(y=PlotIdx, xmin=0, xmax=Dat.shape[0], linewidth=0.5, linestyle='dashed', color=gap_color, zorder=-2)
        ax.vlines(x=np.where(Dat!=4)[0], ymin=PlotIdx-0.5, ymax=PlotIdx+0.5, color=np.array(base_colors)[Dat[np.where(Dat!=4)[0]]], zorder=-1)
        LabelX1.append(L+15)
        LabelX2.append(L+10)
        LabelYMIN.append(PlotIdx-0.5)
        LabelYMAX.append(PlotIdx+0.5)
        ClusterL.append(C)
        PlotIdx += -1
    LabelList = np.array([[i,color,label] for i, (color, label) in enumerate(zip(base_colors, baseName), start=1)], dtype=object)
    for i, (color, label) in enumerate(zip(base_colors, baseName), start=1):
        ax.vlines(x=i, ymin=PlotIdx-100.5, ymax=PlotIdx-99.5, color=color, label=label, zorder=3)
    for ClustI in np.unique(ClusterL):
        ax.vlines(x=ClusterDict[ClustI]+20, ymin=PlotIdx-200-0.5, ymax=PlotIdx-200+0.5, color=category_colors[ClusterDict[ClustI]], linewidth=3, zorder=3, label=ClustI)
    readLabelLen = 0.05 * L 
    for labelIDX in np.arange(len(LabelYMIN)):
        LabelL = LabelYMIN[labelIDX]
        ClustI = ClusterL[labelIDX]
        rect = patches.Rectangle((-readLabelLen, LabelL), readLabelLen-1, 1, linewidth=1, edgecolor=category_colors[ClusterDict[ClustI]], facecolor=category_colors[ClusterDict[ClustI]])
        ax.add_patch(rect)
    ax.set_xlim((-readLabelLen, L+1))
    ax.set_ylim((PlotIdx,1))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    ax.set_title(title)
    return(0)

def SequencePlotFeatureSelect(ax, seqdatamx_Raw, readID, flank_5, flank_3, 
                              title='RawSeq', hcutoff = 3, scutoff=0.05, 
                              base_colors=np.array(['#FFB74D', '#64B5F6', '#81C784', '#F06292']),
                              baseName = np.array(['A','T','C','G']), 
                              gap_color = '#999999', ClusterDict=ClusterDict, 
                              category_colors=category_colors):
    # Plot for feature selection 
    sequences = seqdatamx_Raw
    readLen = np.array([len(x) for x in sequences[1:]])
    DELIDX = np.where(readLen==0)[0]
    if DELIDX.shape[0] > 0:      # Fully DEL reads exist impute alignment matrix as fully gap 
        UnDELIDX = np.setdiff1d(np.arange(len(readIDList)), DELIDX)
        UnDELReads = list(readIDList[UnDELIDX])
        DELReads = list(readIDList[UnDELIDX])
        UnDELSeq = [sequences[I] for I in UnDELIDX]
        unconsensus, unmsa = poa(sequences,1)
        unseqencode_New = list(map(SeqEncoder, unmsa))
        mxlen = len(unseqencode_New[-1])
        readIDList = np.array(UnDELReads + DELReads)     # array in array back
        seqencode_New = np.array(unseqencode_New + [[4] * mxlen] * len(DELReads))
        msa = unmsa + [["-"] * mxlen] * len(DELReads)
    else:
        consensus, msa = poa(sequences,1)
        seqencode_New = np.array(list(map(SeqEncoder, msa)))
    # Remove the Non-associated flank sequence based on reference backbone 
    IDXPool = CallMargin(msa, flank_5, flank_3)
    NonSameIDX = np.setdiff1d(FindNonSameSite(seqencode_New, cutoff=max([hcutoff,seqencode_New.shape[0]*scutoff])), IDXPool)
    SamIDX = np.setdiff1d(np.setdiff1d(np.arange(seqencode_New.shape[1]), NonSameIDX), IDXPool)
    L = np.max([len(x) for x in seqencode_New])
    LabelX1,LabelX2, LabelYMIN, LabelYMAX, ClusterL, CAtypeL = [],[],[],[],[],[]
    PlotIdx = 0 
    for IDX in range(len(seqencode_New)):
        Dat = seqencode_New[IDX]
        C = readID[IDX].split("|")[0]
        if np.where(Dat==4)[0].shape[0] > 0:
            ax.hlines(y=PlotIdx, xmin=0, xmax=Dat.shape[0], linewidth=0.5, linestyle='dashed', color=gap_color, zorder=-2)
        ax.vlines(x=np.where(Dat!=4)[0], ymin=PlotIdx-0.5, ymax=PlotIdx+0.5, color=np.array(base_colors)[Dat[np.where(Dat!=4)[0]]], zorder=-1)
        LabelX1.append(L+15)
        LabelX2.append(L+10)
        LabelYMIN.append(PlotIdx-0.5)
        LabelYMAX.append(PlotIdx+0.5)
        ClusterL.append(C)
        PlotIdx += -1
    FeatureLen = seqencode_New.shape[0] * 0.1
    FeatureLabel = ['Common', 'Flank', 'Features']
    for i, IDXtmp in enumerate([SamIDX, IDXPool, NonSameIDX]):
        ax.vlines(x=IDXtmp, ymin=0, ymax=FeatureLen, color=category_colors[-1*i-1], label=FeatureLabel[i])
    for ClustI in np.unique(ClusterL):
        ax.vlines(x=ClusterDict[ClustI]+20, ymin=PlotIdx-200-0.5, ymax=PlotIdx-200+0.5, color=category_colors[ClusterDict[ClustI]], linewidth=3, zorder=3, label=ClustI)
    readLabelLen = 0.05 * L 
    for labelIDX in np.arange(len(LabelYMIN)):
        LabelL = LabelYMIN[labelIDX]
        ClustI = ClusterL[labelIDX]
        rect = patches.Rectangle((-readLabelLen, LabelL), readLabelLen-1, 1, linewidth=1, edgecolor=category_colors[ClusterDict[ClustI]], facecolor=category_colors[ClusterDict[ClustI]])
        ax.add_patch(rect)
    ax.set_xlim((-readLabelLen, L+1))
    ax.set_ylim((PlotIdx,FeatureLen))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    ax.set_title(title)
    return(0)

def SequencePlotCluster(ax, seqdatamx_Raw, readID, flank_5, flank_3, 
                        title='RawSeq', hcutoff = 3, scutoff=0.05, 
                        base_colors=np.array(['#FFB74D', '#64B5F6', '#81C784', '#F06292']),
                        baseName = np.array(['A','T','C','G']), 
                        gap_color = '#999999', ClusterDict=ClusterDict, 
                        category_colors=category_colors):
    # Plot for filtered matrix 
    sequences = seqdatamx_Raw
    readLen = np.array([len(x) for x in sequences[1:]])
    DELIDX = np.where(readLen==0)[0]
    if DELIDX.shape[0] > 0:      # Fully DEL reads exist impute alignment matrix as fully gap 
        UnDELIDX = np.setdiff1d(np.arange(len(readIDList)), DELIDX)
        UnDELReads = list(readIDList[UnDELIDX])
        DELReads = list(readIDList[UnDELIDX])
        UnDELSeq = [sequences[I] for I in UnDELIDX]
        unconsensus, unmsa = poa(sequences,1)
        unseqencode_New = list(map(SeqEncoder, unmsa))
        mxlen = len(unseqencode_New[-1])
        readIDList = np.array(UnDELReads + DELReads)     # array in array back
        seqencode_New = np.array(unseqencode_New + [[4] * mxlen] * len(DELReads))
        msa = unmsa + [["-"] * mxlen] * len(DELReads)
    else:
        consensus, msa = poa(sequences,1)
        seqencode_New = np.array(list(map(SeqEncoder, msa)))
    # Remove the Non-associated flank sequence based on reference backbone 
    IDXPool = CallMargin(msa, flank_5, flank_3)
    NonSameIDX = np.setdiff1d(FindNonSameSite(seqencode_New, cutoff=max([hcutoff,seqencode_New.shape[0]*scutoff])), IDXPool)
    SamIDX = np.setdiff1d(np.setdiff1d(np.arange(seqencode_New.shape[1]), NonSameIDX), IDXPool)
    seqdatamx = seqencode_New[1:, NonSameIDX]
    readID = readID[1:]
    K, seqdatamx, Rclust, thetap, gamma, pie, BICarray = EMCluster(seqdatamx)
    L = np.max([len(x) for x in seqdatamx])
    LabelX1,LabelX2, LabelYMIN, LabelYMAX, ClusterL, CAtypeL = [],[],[],[],[],[]
    PlotIdx = 0 
    for X in np.unique(Rclust):
        category_tmp = category_colors[X]
        for IDX in np.where(Rclust==X)[0]:
            Dat = seqdatamx[IDX]
            C = readID[IDX].split("|")[0]
            if np.where(Dat==4)[0].shape[0] > 0:
                ax.hlines(y=PlotIdx, xmin=0, xmax=Dat.shape[0], linewidth=0.5, linestyle='dashed', color=gap_color, zorder=-2)
            ax.vlines(x=np.where(Dat!=4)[0], ymin=PlotIdx-0.5, ymax=PlotIdx+0.5, color=np.array(base_colors)[Dat[np.where(Dat!=4)[0]]], zorder=-1)
            LabelX1.append(L+15)
            LabelX2.append(L+10)
            LabelYMIN.append(PlotIdx-0.5)
            LabelYMAX.append(PlotIdx+0.5)
            ClusterL.append(C)
            CAtypeL.append(X)
            PlotIdx += -1
    FeatureLen = seqdatamx.shape[0] * 0.1
    ax.vlines(x=np.arange(seqdatamx.shape[1]), ymin=0, ymax=FeatureLen, color=category_colors[-3], label='Features')
    for ClustI in np.unique(ClusterL):
        ax.vlines(x=ClusterDict[ClustI]+20, ymin=PlotIdx-200-0.5, ymax=PlotIdx-200+0.5, color=category_colors[ClusterDict[ClustI]], linewidth=3, zorder=3, label=ClustI)
    for CAI in np.unique(Rclust):
        ax.vlines(x=CAI+20, ymin=PlotIdx-200-0.5, ymax=PlotIdx-200+0.5, color=category_colors[-1*CAI-1], linewidth=3, zorder=3, label='Cluster %s' % CAI)
    readLabelLen = 0.05 * L 
    for labelIDX in np.arange(len(LabelYMIN)):
        LabelL = LabelYMIN[labelIDX]
        ClustI = ClusterL[labelIDX]
        CAI = CAtypeL[labelIDX]
        rect = patches.Rectangle((-readLabelLen, LabelL), readLabelLen-1, 1, linewidth=1, edgecolor=category_colors[ClusterDict[ClustI]], facecolor=category_colors[ClusterDict[ClustI]])
        ax.add_patch(rect)
        rect2 = patches.Rectangle((-2*readLabelLen, LabelL), readLabelLen-1, 1, linewidth=1, edgecolor=category_colors[-1*CAI-1], facecolor=category_colors[-1*CAI-1])
        ax.add_patch(rect2)
    ax.set_xlim((-2*readLabelLen, L+1))
    ax.set_ylim((PlotIdx,FeatureLen))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    ax.set_title(title)
    return(Rclust, readID)

def barhmaker(ax, Rclust, readID, LabelList,category_colors=category_colors,ClusterDict=ClusterDict):
    # the component of each cluster  as barh plot 
    Nlabel,Tlabel = LabelList
    clusterType = np.unique(Rclust)
    valueT = np.zeros((clusterType.shape[0],))
    valueN = np.zeros((clusterType.shape[0],))
    readLabel = np.array([R.split("|")[0] for R in readID])
    for C in clusterType:
        tmpLabel = readLabel[np.where(Rclust==C)[0]]
        valueT[np.where(clusterType==C)[0]] += np.where(tmpLabel==Tlabel)[0].shape[0]
        valueN[np.where(clusterType==C)[0]] += np.where(tmpLabel==Nlabel)[0].shape[0]
    ax.barh(['Cluster %s' % x for x in clusterType], valueT, label='Tumor ReadNum',color=category_colors[ClusterDict[LabelList[-1]]])
    ax.barh(['Cluster %s' % x for x in clusterType], valueN, left=valueT, label='Normal ReadNum', color=category_colors[ClusterDict[LabelList[0]]])
    ax.set_title('Componenet Analysis')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    return(0)    

def GetNodeID(seqMX, threshold=10, NodeIDStart=0):
    # let difference lower than 50 considered as same node , -1 represent for del, if smaller than 10 sequence is not 4 as gap 
    NodeLabel = np.zeros(seqMX.shape[0]) + NodeIDStart+1
    currentNodeID = NodeIDStart+1
    for i in np.arange(seqMX.shape[0]):
        if (np.where(seqMX[i, :] != 4)[0].shape[0]<threshold) or (np.where(seqMX[i, :] == 4)[0].shape[0]>=0.8*seqMX.shape[1]):
            NodeLabel[i] = -1
    if np.where(NodeLabel!=-1)[0].shape[0] <= 1:
        return(list(NodeLabel))
    else:
        TMPIDX = list(np.where(NodeLabel!=-1)[0])
        finishedIDX = [TMPIDX[0]]
        while len(finishedIDX) != len(TMPIDX):
            j = np.setdiff1d(TMPIDX, finishedIDX)[0]
            state = 0
            for i in finishedIDX:
                if np.where(seqMX[j,:]!=seqMX[i,:])[0].shape[0]<threshold:
                    finishedIDX.append(j)
                    NodeLabel[j] = NodeLabel[i]
                    state = 1
            if state == 0:
                currentNodeID += 1
                finishedIDX.append(j)
                NodeLabel[j] = currentNodeID
    return(list(NodeLabel))

def ShowConsensus(ax, seqdatamx_Raw, Rclust, readID, flank_5, flank_3, LabelList, 
                  title='Local Graph Opptimization', hcutoff = 3, scutoff=0.05, 
                  base_colors=np.array(['#FFB74D', '#64B5F6', '#81C784', '#F06292']),
                  baseName = np.array(['A','T','C','G']), 
                  gap_color = '#999999', ClusterDict=ClusterDict, 
                  category_colors=category_colors):
    # represent SV as local graph 
    reference = seqdatamx_Raw[0]
    consensusList = [reference]
    labelRecord = ['ref']
    for R in np.unique(Rclust):
        readIDXList = np.where(Rclust==R)[0]
        readLabel = np.array([x.split("|")[0] for x in readID[1:]])[readIDXList]
        if np.where(readLabel==LabelList[-1])[0].shape[0] == readLabel.shape[0]:
            labelRecord.append('Cluster %s(somatic)' % R)
        else:
            labelRecord.append('Cluster %s(germline)' % R)
        consensus, msa = poa([seqdatamx_Raw[I] for I in readIDXList],1)
        consensusList.append(consensus)
    c_total,msa_total = poa(consensusList,1)
    seqencode_New = np.array(list(map(SeqEncoder, msa_total)))
    IDXPool_f5, IDXPool_f3 = CallMargin_f5(msa_total, flank_5), CallMargin_f3(msa_total, flank_3)
    borderstart, borderend = IDXPool_f5[-1]+1, IDXPool_f3[0]
    commonIDX = np.array([x for x in np.arange(seqencode_New.shape[1]) if (x >=borderstart) and (x<borderend) and (np.unique(seqencode_New[:,x]).shape[0]==1)])
    commonblock = [x for x in np.split(commonIDX, np.where(np.diff(commonIDX)>=10)[0]+1) if x.shape[0]>=10]
    uncommonIDX = np.array([x for x in np.arange(seqencode_New.shape[1]) if (x >=borderstart) and (x<borderend) and (np.unique(seqencode_New[:,x]).shape[0]>1)])
    uncommonblock = [x for x in np.split(uncommonIDX, np.where(np.diff(uncommonIDX)>=10)[0]+1) if x.shape[0]>=10]
    tmpDf = pd.DataFrame([[x[0], x[-1], 'commonBlock'] for x in commonblock] + [[x[0], x[-1], 'uncommonBlock'] for x in uncommonblock], columns=['start', 'end', 'label']).sort_values(['start'], ascending=True)
    tmpDf.index = np.arange(1, tmpDf.shape[0]+1)
    NodeCollect = [[0]*len(msa_total)]    # flank 5 
    NodeIDStart = 0
    for i,rows in tmpDf.iterrows():
        if rows['label'] == 'commonBlock':
            NodeCollect.append([NodeIDStart+1]*len(msa_total))
            NodeIDStart+=1
        else:
            seqMX = seqencode_New[:,rows['start']:rows['end']]
            NodeLabel = GetNodeID(seqMX, NodeIDStart=NodeIDStart)
            NodeIDStart = np.max([NodeIDStart, np.max(NodeLabel)])
            NodeCollect.append(NodeLabel)
    NodeCollect.append([NodeIDStart+1]*len(msa_total))
    G = nx.DiGraph()
    NodeIDs = np.unique(np.concatenate(NodeCollect))
    NodeList = NodeIDs[np.where(NodeIDs!=-1)[0]]
    for i in range(np.array(NodeCollect).shape[1]):
        trace = np.array(NodeCollect)[:,i]
        L = labelRecord[i]
        i = 0
        j = 1
        while (i<len(trace)-1) and (j<len(trace)):
            if (trace[i]!=-1) and (trace[j]!=-1):
                G.add_edge(int(trace[i]), int(trace[j]))
                i = j
                j += 1
            elif (trace[j]==-1):
                j += 1
                continue
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, ax=ax, with_labels=True, arrows=True)
    ax.set_title(title)
    return(np.array(NodeCollect))
        
# *         Function finished            *#
# *          Main Function               *#

def DrawPipe(TDRecord, refFile,bamFileList, LabelList, saveDir, offset,mapQ, graph=False):
    # Visualize Data 
    TDchrom, TDStart, TDEnd = TDRecord.strip().split("\t")[0],int(TDRecord.strip().split("\t")[1]), int(TDRecord.strip().split("\t")[2])
    ref = pysam.FastaFile(refFile)
    refSeq = ref.fetch(TDchrom, TDStart-offset, TDEnd+offset).upper()
    flank_5, flank_3 = ref.fetch(TDchrom, TDStart-offset, TDStart).upper(), ref.fetch(TDchrom, TDEnd, TDEnd+offset).upper()
    readTDSeq, readIDList, FlankMQ = FetchTDsubSeq(refFile, bamFileList, LabelList, TDRecord)
    CertainIDX = [I for I in range(len(FlankMQ)) if np.min(FlankMQ[I])>=mapQ]
    readTDSeqNew = [readTDSeq[I] for I in CertainIDX]                                                # sub read selection uncertain flank aln reads would be removed
    ReadIDs = [readIDList[I] for I in CertainIDX]
    seqdatamx_Raw = [refSeq] + readTDSeqNew
    readID = ['ref|Seq'] + ReadIDs
    # Make Figure 
    fig = plt.figure(figsize=(12,12))
    gs = GridSpec(3, 2, width_ratios=[1,1],height_ratios=[1,1,1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    SequencePlotRaw(ax1, seqdatamx_Raw, readID, 'RawData')
    consensus, msa = poa(seqdatamx_Raw,1)
    SequencePlotRaw(ax2, msa, readID, 'sPOA')
    SequencePlotFeatureSelect(ax3, seqdatamx_Raw, readID, flank_5, flank_3, 'FeatureSelection')
    Rclust, readID = SequencePlotCluster(ax4, seqdatamx_Raw, readID, flank_5, flank_3, 'Clustering')
    barhmaker(ax5, Rclust, readID, LabelList)
    if graph:
        ShowConsensus(ax6, seqdatamx_Raw, Rclust, ['ref|seq']+readID, flank_5, flank_3, LabelList)
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir, '_'.join(TDRecord.strip().split("\t"))+".Visualize.png"), dpi=600) 
    plt.close()
    return(TDRecord.split("\t"))

def main(args):
    logging.info('Start working')
    start_time = time.time()
    # somatic repeat calling software main program 
    ## First check input parameters 
    tumorbamList = args.Tumorbam.split(",")
    normalbamList = args.Normalbam.split(",")
    TsampleID = args.TSampleID.split(",")
    NsampleID = args.NSampleID.split(",")
    offset = int(args.offset)
    mapQ = int(args.mapQ)
    if len(TsampleID) != len(tumorbamList):
        print("SampleID not meet tumor bam file, exit !")
        exit(1)
    if len(NsampleID) != len(normalbamList):
        print("SampleID not meet normal bam file, exit !")
        exit(1)
    bamFileList = tumorbamList + normalbamList
    LabelList = ["tumor" for x in TsampleID]  + ['normal' for x in NsampleID]
    rawoutput = '%s.vs.%s.TandemRepeat.Raw.bed' % ("-".join(TsampleID), '-'.join(NsampleID))
    file_path = os.path.join(args.savedir, rawoutput)
    with open(args.windowBed) as bedin:
        TDRecordList = ["\t".join(x.strip().split("\t")[0:3]) for x in bedin.readlines()]
    DrawPipe_exe = functools.partial(DrawPipe, refFile=args.Reference, bamFileList=bamFileList, LabelList=LabelList, offset=offset, mapQ=mapQ, graph=args.graph, saveDir=args.savedir)
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    ## Run program 
    P = Pool(processes=int(args.thread))
    results = []
    for TDRecord in TDRecordList:
        result = P.apply_async(DrawPipe_exe, (TDRecord,))
        results.append(result)
    with open(os.path.join(args.savedir, rawoutput), 'w') as f:
        outRecord = 0
        while results:
            for result in results:
                if result.ready():
                    output = result.get()
                    f.write("\t".join([str(x) for x in output]) + '\n')
                    f.flush()
                    results.remove(result)
                    outRecord += 1
    os.system('sort -k1,1 -k2,2n {outDir}/{rawoutput} -o {outDir}/{rawoutput}'.format(outDir=args.savedir, rawoutput=rawoutput))
    time_span = (time.time() - start_time) / 3600
    logging.info(f'work finished with {time_span} hour')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-w", "--windowBed", required=True, help="pre made tandem repeat windows chrom,repeatStart,repeatEnd")
    parser.add_argument("-T", "--Tumorbam", required=True, help="ONT read alignment bam file, multiple bam should seperated with ','")
    parser.add_argument("-N", "--Normalbam", required=True, help="ONT read alignment bam file, multiple bam should seperated with ','")
    parser.add_argument("-t", "--TSampleID", required=True, help="SampleID for this somatic project, if multiple sample should seperated with ',' and have same length with tumor bam")
    parser.add_argument("-n", "--NSampleID", required=True, help="SampleID for this somatic project, if multiple sample should seperated with ',' and have same length with normal bam")
    parser.add_argument("-r", "--Reference", required=True, help="reference file fasta path")
    parser.add_argument("-s", "--savedir", required=True, help="dir for result file save")
    parser.add_argument("-p", "--thread", required=True, help="CPU use for program")
    parser.add_argument("-o", "--offset", type=int, default=50, help="offset default value is 50")
    parser.add_argument("-q", "--mapQ", type=int, default=5, help="mapQ default value is 5")
    parser.add_argument('--graph', action='store_true', help="Set visualization show graph genome in the last figure, may take much longer time to do it, by default False")
    args = parser.parse_args()
    main(args)

# *     Main Function finished           *#
# *             Data Test                 *# 

# TDRecord = 'chr6\t601190\t602485'
# TDchrom, TDStart, TDEnd = TDRecord.strip().split("\t")[0],int(TDRecord.strip().split("\t")[1]), int(TDRecord.strip().split("\t")[2])
# bamFileList = ["/mnt/data2/tkl/1395/benchmarkCellLine_Evaluation/HCC1395BL.MindaSV.bam",
#                "/mnt/data2/tkl/1395/benchmarkCellLine_Evaluation/HCC1395.MindaSV.bam"]
# LabelList = ['normal', 'tumor']
# refFile = "/mnt/data2/tkl/genome/mainChr_hg38_UCSC.fa"
# readTDSeq, readIDList, FlankMQ = FetchTDsubSeq(refFile, bamFileList, LabelList, TDRecord)
# base_colors = np.array(['#FFB74D', '#64B5F6', '#81C784', '#F06292'])
# baseName = np.array(['A','T','C','G'])
# gap_color = '#999999'
# ref = pysam.FastaFile(refFile)
# refSeq = ref.fetch(TDchrom, TDStart-200, TDEnd+200).upper()
# flank_5, flank_3 = ref.fetch(TDchrom, TDStart-200, TDStart).upper(), ref.fetch(TDchrom, TDEnd, TDEnd+200).upper()


# seqdatamx_Raw = [refSeq] + readTDSeq
# L = np.max([len(x) for x in seqdatamx_Raw])
# LabelX1,LabelX2, LabelYMIN, LabelYMAX, ClusterL, CAtypeL = [],[],[],[],[],[]
# readID = ['ref|Seq'] + readIDList

# fig = plt.figure(figsize=(12,12))
# gs = GridSpec(3, 2, width_ratios=[1,1],height_ratios=[1,1,1])
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[1, 0])
# ax4 = fig.add_subplot(gs[1, 1])
# ax5 = fig.add_subplot(gs[2, 0])
# ax6 = fig.add_subplot(gs[2, 1])

# SequencePlotRaw(ax1, seqdatamx_Raw, readID, 'RawData')
# consensus, msa = poa(seqdatamx_Raw,1)
# SequencePlotRaw(ax2, msa, readID, 'sPOA')
# SequencePlotFeatureSelect(ax3, seqdatamx_Raw, readID, flank_5, flank_3, 'FeatureSelection')
# Rclust, readID = SequencePlotCluster(ax4, seqdatamx_Raw, readID, flank_5, flank_3, 'Clustering')
# barhmaker(ax5, Rclust, readID, LabelList)
# ShowConsensus(ax6, seqdatamx_Raw, Rclust, ['ref|seq']+readID, flank_5, flank_3, LabelList)
# plt.tight_layout()
# plt.savefig()   # 存储图表。
