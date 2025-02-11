'''
__Author__: Kailing Tu
__Version__: v18.0.0
__ReleaseTime__: 2024-03-22

ReleaseNote:
    Change FetchTDsubSeq. Now we are going to consider both supplementary and secondary alignment reads, the aim of FetchTDsubSeq is to find the 5'Flank start and 3'Flank end for each reads with primary alignment in TD region
    Change spoa graph maker. Now spoa graph is maked by simi-global alignment rather than global alignment. 
    Undo things: feature selection, make less feature for reads by merging same feature together. 
    
Requirement:
    pysam v0.19.1
    spoa
    numpy v1.21.5
    pandas v1.3.4
    mappy v2.20
Description: Parsing ONT bam file to find aim read sequence and filter out low quality reads for further analysis
    Modular components:
        FetchTDsubSeq:
            Function fetch sub sequence of each read around Tandem duplication region. 
            Primary alignment read sequence with 5' and 3' flank sequence according to pairwise alignment data would be selected 
        SeqAligner:
            Function for multiple sequence alignment matrix generation using spoa package 
        SeqEncoder:
            Function for read sequence alignment matrix encoding 
        SeqDecoder:
            Function to decode 0,1,2,3,4 MSA matrix to A,T,C,G,- sequence matrix and for each reads we remove gap
        CallMargin:
            Function to find reference 5' and 3' flank sequence in MSA matrix 
        FindNonSameSite:
            Function filter out low difference site in seqdatamx 
            By default second largest base < max([3, 0.1*N]) site will be removed
            Further, it will remove regions matched to 5' and 3' flank sequence based on hg38 reference (defined by Call Margin)
        DataMaker:
            Function to scan bam file according to the TD bed record and filter proper sub reads for further analysis
'''
import pandas as pd 
from spoa import poa
import pysam 
import numpy as np
import os,re
import sqlite3
from concurrent.futures import ProcessPoolExecutor
import logging
import functools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Local data scanner 
def reverse_complement(sequence):
    # reverse_complement sequence 
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    reverse_sequence = sequence[::-1]
    reverse_complement_sequence = ''.join([complement[base] for base in reverse_sequence])
    return(reverse_complement_sequence)

def ReadsLoci(reads, start, end, offset=0):
    # input reads, F5,F3 parameter get read Loci
    # reads should be selected by 
    aln_pair = np.array(reads.aligned_pairs)
    aln_pair_linear = aln_pair[np.where((aln_pair[:,0]!=None)&(aln_pair[:,1]!=None))]
    startPosIDX = np.where(aln_pair_linear[:,1]<=start)[0][-1]
    endPosIDX = np.where(aln_pair_linear[:,1]>=end)[0][0]
    readStart, readEnd = offset + aln_pair_linear[startPosIDX, 0], offset + aln_pair_linear[endPosIDX, 0]
    return([readStart, readEnd])        

def FetchTDsubSeq(refFile, bamFileList, LabelList, TDRecord, offset=200):
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
                    CIGAR = np.array(reads.cigar)
                    if CIGAR[0][0] == 5:
                        offset = CIGAR[0][1]
                F5_readsIDX.append([reads.qname] +  ReadsLoci(reads, F5start, F5end, offset))
                F5_readName.append(reads.qname)
            # Work For F3 flank 
            if (reads.reference_start<F3start) and (reads.reference_end>F3end) and (not reads.is_secondary):
                offset = 0
                if reads.is_supplementary:
                    CIGAR = np.array(reads.cigar)
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
    consensus, msa = poa(seqList)
    seqdatamx = list(map(SeqEncoder, msa))
    return(seqdatamx)

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

def MSAFeatureSelection(sequenceList, flank_5, flank_3, readIDList, hcutoff=3, scutoff=0.05):
    '''
    Make MAS and select features for MSA matrix 
    -*-: Fixed full read DEL stage, to avoid such reads drop out from poa process, we set DELIDX if loop to filter out these reads and finally add it back to the bottom of seqencode_New with full gap data. (Kailing Tu at 2024-08-19)
    :param:
        sequenceList: list object with reference sequence at the first position, other read subsequence follows
        flank_5: 5' flank sequence with all upper str generated from reference genome 
        flank_3: 3' flank sequence with all upper str generated from reference genome
        readIDList: read ID list for all reads with labels 
        hcutoff : second frequency feature number cutoff, default 3 
        scutoff : second frequency feature number percentage cutoff, default 0.05
    :return:
        seqencode_New: MSA aligned and re-encoded matrix for visualization 
        seqdatamx: MSA aligned, re-encoded and feature selected matrix for EM clustering pipeline
        readIDList: np.array for readIDs with label, order would be changed if there is full DEL reads sequence. (Kailing Tu at 2024-08-19)
    '''
    # spoa graph making
    sequences = sequenceList
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
    TDseq_Raw = seqencode_New[1:,np.setdiff1d(np.arange(seqencode_New.shape[1]), IDXPool)]
    # remove variant sequence features 
    seqdatamx = TDseq_Raw[:,FindNonSameSite(TDseq_Raw, cutoff=max([hcutoff,seqencode_New.shape[0]*scutoff]))]
    return(seqencode_New, seqdatamx, readIDList)

def DataMaker(TDRecord, refFile, bamFileList, LabelList, offset=200,mapQ=5):
    # input reference, bamfilelist, label, TDrecord
    # selecting reads and feature 
    # output seqdatamx for further clustering 
    # raw sequence generation 
    flag = 'NormalOutput'
    readTDSeq, readIDList, FlankMQ = FetchTDsubSeq(refFile, bamFileList, LabelList, TDRecord, offset=offset)
    CertainIDX = [I for I in range(len(FlankMQ)) if np.min(FlankMQ[I])>=mapQ]
    refFasta = pysam.FastaFile(refFile)
    chrom,start,end = TDRecord.strip().split("\t")[0:3]
    flank_5,flank_3 = refFasta.fetch(chrom, int(start)-offset, int(start)).upper(), refFasta.fetch(chrom, int(end),int(end)+offset).upper()
    exampleSequence = refFasta.fetch(chrom, int(start)-offset, int(end)+offset).upper()
    if re.search('N', flank_5) or re.search('N', flank_3) or re.search('N', exampleSequence):
            sequenceList = np.array([])
            ReadIDs = np.array([])
            flag = "GapRegion"
    elif (len(CertainIDX) <= 3):
        sequenceList = np.array([])
        ReadIDs = np.array([])
        flag = 'NoEnoughspanReads'
    else:
        # datafilter remove imcomplete flank sequence reads 
        readTDSeqNew = [readTDSeq[I] for I in CertainIDX]                                                # sub read selection uncertain flank aln reads would be removed
        ReadIDs = np.array([readIDList[I] for I in CertainIDX])
        sequenceList = [refFasta.fetch(chrom, int(start)-offset, int(end)+offset).upper()] + readTDSeqNew
    return(sequenceList, ReadIDs, flank_5,flank_3, TDRecord, flag)

def ReadsLoci2(reads, start,end, offset):
    aln_pair = np.array(reads.aligned_pairs)
    aln_pair_linear = aln_pair[np.where((aln_pair[:,0]!=None)&(aln_pair[:,1]!=None))]
    if (reads.reference_start<start) and (reads.reference_end>end):                                       # --|--|--
        startPosIDX = np.where(aln_pair_linear[:,1]<=start)[0][-1]
        endPosIDX = np.where(aln_pair_linear[:,1]>=end)[0][0]
    elif (reads.reference_start >= start) and (reads.reference_start<end) and (reads.reference_end>end):  # | --|--
        startPosIDX = 0
        endPosIDX = np.where(aln_pair_linear[:,1]>=end)[0][0]
    elif (reads.reference_start<start) and (reads.reference_end>start) and (reads.reference_end<=end):    # --|-- |
        startPosIDX = np.where(aln_pair_linear[:,1]<=start)[0][-1]
        endPosIDX = -1
    elif (reads.reference_start>=start) and (reads.reference_end<=end):                                   # | -- |
        startPosIDX = 0
        endPosIDX = -1
    readStart, readEnd = offset + aln_pair_linear[startPosIDX, 0], offset + aln_pair_linear[endPosIDX, 0]
    return([readStart, readEnd])

def SubSeqInWindow(bamFileList, LabelList, window):
    # input bamFile List , LabelList reference window get subseq of reads 
    chrom, Start, End = window.strip().split("\t")[0],int(window.strip().split("\t")[1]), int(window.strip().split("\t")[2])
    readIDList, readTDSeq, FlankMQ = [],[],[]   # readID, readSequence, primary ALN mapQ
    tmpReadSeqArr, readInfoRecord = [], []
    for bamIDX in range(len(bamFileList)):
        for reads in pysam.AlignmentFile(bamFileList[bamIDX]).fetch(chrom,Start,End):
            offset = 0 
            if not(reads.is_secondary or reads.is_supplementary):             # reads should have primary alignment in the window to get the whole sequence 
                tmpReadSeqArr.append([LabelList[bamIDX] + "|" + reads.query_name, reads.query_sequence, reads.mapq])
            if not reads.is_secondary:                                        # remove secondary alignment to avoid chaos of sequence 
                CIGAR = np.array(reads.cigar)
                if CIGAR[0][0] == 5:
                    offset = CIGAR[0][1]
                readInfoRecord.append([LabelList[bamIDX] + "|" + reads.query_name] + ReadsLoci2(reads, Start, End, offset))
    SeqDf = pd.DataFrame(tmpReadSeqArr, columns=['readID', 'sequence', 'mapq'])
    SeqDf.index = SeqDf['readID']
    InfoDf = pd.DataFrame(readInfoRecord, columns = ['readID', 'readStart', 'readEnd']).sort_values(['readStart'],ascending=True)
    spanReadIDs = np.intersect1d(np.array(SeqDf['readID']), np.unique(InfoDf['readID']))
    for IDs in spanReadIDs:
        InfoDf_Sub = InfoDf.loc[InfoDf['readID']==IDs]
        readSeq = SeqDf.loc[IDs, 'sequence']
        insideSeq = ''
        for i,rows in InfoDf_Sub.iterrows():
            insideSeq += readSeq[rows['readStart']:rows['readEnd']]
        readIDList.append(IDs)
        readTDSeq.append(insideSeq)
        FlankMQ.append(SeqDf.loc[IDs, 'mapq'])
    return(readTDSeq, readIDList, FlankMQ)

def DataMaker2(TDRecord, refFile, bamFileList, LabelList, offset=200,mapQ=5):
    # remake window for DUP record get read sequence from 5' and 3' breakpoint location 
    # remake data would happen if and only if TDRecord is DUP and output result is negative, we gonna re calculate it at the corner of duplication window to check whether it is somaitc event 
    refFasta = pysam.FastaFile(refFile)
    flag5,flag3 = 'UnspanedSV', 'UnspannedSV'
    chrom,start,end = TDRecord.strip().split("\t")[0:3]
    window_flank5 = "\t".join([chrom, start, str(int(start)+50)])
    window_flank3 = "\t".join([chrom, str(int(end)-50), end])
    readTDSeq_5, readIDList_5, FlankMQ_5 = SubSeqInWindow(bamFileList, LabelList, window_flank5)
    readTDSeq_3, readIDList_3, FlankMQ_3 = SubSeqInWindow(bamFileList, LabelList, window_flank3)
    CertainIDX5 = [I for I in range(len(FlankMQ_5)) if np.min(FlankMQ_5[I])>=mapQ]
    if (len(CertainIDX5) <= 3):
        flag5 = 'Unspaned+NotEnoughReads'
        sequenceList_5 = np.array([])
        ReadIDs_5 = np.array([])
    else:
        readTDSeqNew_5 = [readTDSeq_5[I] for I in CertainIDX5]                                                # sub read selection uncertain flank aln reads would be removed
        ReadIDs_5 = np.array([readIDList_5[I] for I in CertainIDX5])
        sequenceList_5 = [refFasta.fetch(chrom, int(start), int(start)+50).upper()] + readTDSeqNew_5
    CertainIDX3 = [I for I in range(len(FlankMQ_3)) if np.min(FlankMQ_3[I])>=mapQ]
    if (len(CertainIDX3) <= 3):
        flag3 = 'Unspaned+NotEnoughReads'
        sequenceList_3 = np.array([])
        ReadIDs_3 = np.array([])
    else:
        readTDSeqNew_3 = [readTDSeq_5[I] for I in CertainIDX3]                                                # sub read selection uncertain flank aln reads would be removed
        ReadIDs_3 = np.array([readIDList_5[I] for I in CertainIDX3])
        sequenceList_3 = [refFasta.fetch(chrom, int(end)-50, int(end)).upper()] + readTDSeqNew_3
    return([[sequenceList_5, ReadIDs_5, '','', TDRecord, flag5],[sequenceList_3, ReadIDs_3, '','', TDRecord, flag3]])

# Whole genome data scanner 
def makeupDB(bed_file,dbName, batchsize=500000):
    conn = sqlite3.connect('%s.sqlite' % dbName)
    cursor = conn.cursor()
    # 创建调整后的表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reads_length (
        read_id TEXT PRIMARY KEY,
        length INTEGER
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reads_alignment (
        id INTEGER PRIMARY KEY,
        read_id TEXT,
        chrom TEXT,
        start INTEGER,
        end INTEGER,
        mapQ INTEGER,
        strand TEXT,
        FOREIGN KEY (read_id) REFERENCES reads_length (read_id)
    )
    ''')
    # 创建索引
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_read_id ON reads_alignment (read_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_read_id ON reads_length (read_id)')
    conn.commit()
    with pysam.TabixFile(bed_file) as tbx:
        batch_insert_data = []
        RecordNum = 0
        for row in tbx.fetch():
            # 将每行数据添加到列表中
            rows = row.split("\t")
            batch_insert_data.append((rows[3], rows[0], int(rows[1]), int(rows[2]), rows[4], rows[5]))
            # 每累积一定数量的数据后执行一次批量插入
            if len(batch_insert_data) >= batchsize:  # 例如，每10w行为一个批次
                cursor.executemany('''
                INSERT INTO reads_alignment (read_id, chrom, start, end, mapQ, strand)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', batch_insert_data)
                conn.commit()
                batch_insert_data = []  # 清空列表，为下一个批次准备
                RecordNum += batchsize
                logging.info(f"{RecordNum} alignment record insert to reads_alignment table")
        # 插入剩余的数据
        if batch_insert_data:
            cursor.executemany('''
            INSERT INTO reads_alignment (read_id, chrom, start, end, mapQ, strand)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', batch_insert_data)
            conn.commit()
            RecordNum += len(batch_insert_data)
            logging.info(f"reads_alignment table finished data insertion with {RecordNum} alignment record inserted")
        else:
            logging.info(f"reads_alignment table finished data insertion with {RecordNum} alignment record inserted")
        RecordNum = 0
    cursor.close()
    conn.close()
    return('%s.sqlite' % dbName)

# find reads 
def query_reads(dbFile, read_id):
    conn = sqlite3.connect(dbFile)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reads_alignment WHERE read_id = ?', (read_id,))
    alignment_info = cursor.fetchall()
    cursor.close()
    conn.close()
    return(alignment_info)

# Read spanChrom percentage 
def spanchrRatio(readIDList, dbFile):
    ALNInfoWhole = []
    for readID in readIDList:
        ALNInfoWhole.append(query_reads(dbFile, readID))
    dfread_Whole = pd.DataFrame(np.vstack(ALNInfoWhole), columns=['id', 'read_id', 'chrom', 'start','end', 'mapQ', 'strand'])
    dfread_Whole_group = np.array(dfread_Whole.groupby(['read_id'])['chrom'].apply(lambda x: np.unique(x).shape[0]))
    spanChromRate = np.where(dfread_Whole_group!=1)[0].shape[0] / dfread_Whole_group.shape[0] if dfread_Whole_group.shape[0] > 0 else np.nan
    return(spanChromRate)

# Background information maker: 
def OVLEN(window, start,end):
    chrom,ws,we = window.strip().split("\t")[0:3]
    ws,we = int(ws), int(we)
    if (start <= ws) and (end>=we):       # --|--|--
        return(we-ws)
    elif (start>ws) and (end<we):         # |- |
        return(end-start)
    elif (start>ws) and (end>we):         # | -|--
        return(we-start)
    elif (start<ws) and (end < we):       # --|- | 
        return(end-ws)
    else:
        return(0)

def windowInfo(window, bed_file,dbFile, mapQcutoff=5,showchromSpan=False, showmapQ=False):
    # detect read Number of each imput window in chrom\tstart\tend format 
    # detect read mapQ of each input window 
    logging.info(f"Work on window {window}")
    chrom,start,end = window.strip().split("\t")[0:3]
    window_write = chrom + "_" + start + "_" + end
    windowLen = int(end) - int(start)
    readList = []
    with pysam.TabixFile(bed_file) as tbx:
        for reads in tbx.fetch(chrom,int(start), int(end)):
            readList.append(reads.strip().split("\t")[0:6])
    readDf = pd.DataFrame(readList, columns=['chrom', 'start','end','readID','mapQ', 'strand'])
    for col in ['start', 'end', 'mapQ']:
        readDf[col] = readDf[col].apply(int)
    readDf_group = pd.concat([readDf.groupby('readID')['chrom'].apply(lambda x: list(x)[0]), 
                              readDf.groupby('readID')['start'].apply(lambda x: np.min(x)),
                              readDf.groupby('readID')['end'].apply(lambda x: np.max(x)),
                              readDf.groupby('readID')['mapQ'].apply(lambda x: np.min(x)),
                              readDf.groupby('readID')['strand'].apply(lambda x: list(x)[0])], axis=1)
    if readDf_group.shape[0] > 0:
        readDf_group['covLen'] = readDf_group.apply(lambda x: OVLEN(window, x['start'], x['end']), axis=1)
        MAPQ_Rate = readDf_group.loc[readDf_group['mapQ']<mapQcutoff].shape[0] / readDf_group.shape[0] if readDf_group.shape[0]!=0 else np.nan
        COV_Rate = np.sum(readDf_group['covLen']) / windowLen
        readIDList = list(readDf_group.index)
        if showchromSpan:
            spanChromRate = spanchrRatio(readIDList, dbFile)
            logging.info(f"Finished Work on window {window}")
            return([window_write, COV_Rate, MAPQ_Rate, spanChromRate, ",".join(readIDList)])
        else:
            logging.info(f"Finished Work on window {window}")
            return([window_write, COV_Rate, MAPQ_Rate])
    else:
        COV_Rate, MAPQ_Rate, spanChromRate = np.nan,np.nan,np.nan
        if showchromSpan:
            logging.info(f"Finished Work on window {window}")
            return([window_write, COV_Rate, MAPQ_Rate, spanChromRate, ''])
        else:
            logging.info(f"Finished Work on window {window}")
            return([window_write, COV_Rate, MAPQ_Rate])    

def background(windowFile, bed_file, dbFile,showchromSpan=False,workthread=100):
    with open(windowFile) as input:
        windowList = input.readlines()
    if showchromSpan:
        windowList = [x for x in windowList if re.search(r'EMOutput', x)]
    windowInfo_exe = functools.partial(windowInfo, bed_file=bed_file, dbFile=dbFile, showchromSpan=showchromSpan)
    with ProcessPoolExecutor(max_workers=int(workthread)) as executor:
        bg = list(executor.map(windowInfo_exe, windowList))
    if showchromSpan:
        df = pd.DataFrame(bg,columns=['window', 'COV', 'mapQRate', 'chromSpan', 'TotalReadID'])
    else:
        df = pd.DataFrame(bg,columns=['window', 'COV', 'mapQRate'])
    return(df)

