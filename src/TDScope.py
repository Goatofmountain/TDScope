import os,re
import pandas as pd 
import numpy as np 
import pysam 
import argparse
from DecisionMaker import *
from DataScanner import *
import PairwiseCompare
from SomTDDetector import TDscope
from OutVCF import *
import functools
from multiprocessing import Pool
import time
import logging
import gzip
import sqlite3
from joblib import dump, load
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Support Functions

# Major functions 
def localGraph(args):
    # Handle the logic for the localGraph module
    logging.info('Local Graph : Start working')
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
    LabelList = [x+"_tumor" for x in TsampleID]  + [x+'_normal' for x in NsampleID]
    rawoutput = '%s.vs.%s.TandemRepeat.Raw.bed' % ("-".join(TsampleID), '-'.join(NsampleID))
    file_path = os.path.join(args.savedir, rawoutput)
    with open(args.windowBed) as bedin:
        TDRecordList = ["\t".join(x.strip().split("\t")) for x in bedin.readlines()]
    Decision_exe = functools.partial(Decision, Tlabel='tumor', readcutoff=3, hcutoff=3, scutoff=0.05)
    DataMaker_exe = functools.partial(DataMaker, refFile=args.Reference, bamFileList=bamFileList, LabelList=LabelList, offset=offset,mapQ=mapQ)
    DataMaker2_exe = functools.partial(DataMaker2, refFile=args.Reference, bamFileList=bamFileList, LabelList=LabelList, offset=offset,mapQ=mapQ)
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    ## Run program 
    if int(args.thread) > 6:
        P = Pool(processes=6)
    else:
        P = Pool(processes=int(args.thread))
    results = []
    for TDRecord in TDRecordList:
        result = P.apply_async(TDscope, (TDRecord,DataMaker_exe,DataMaker2_exe,Decision_exe,))
        results.append(result)
    with open(file_path, 'w') as f:
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
    logging.info(f'Local Graph : work finished with {time_span} hour')
    return(file_path)

def AlnFeature(args):
    # Handel the logic for the AlnFeature module 
    logging.info('Alignment feature collection module : Start working')
    if not os.path.exists(args.savedir):
        os.system('mkdir -p %s' % args.savedir)
    TsampleID = args.TSampleID
    NsampleID = args.NSampleID
    TbedFile = os.path.join(args.savedir, '%s.bed.gz' % TsampleID)
    NbedFile = os.path.join(args.savedir, '%s.bed.gz' % NsampleID)
    if not os.path.exists(TbedFile):
        os.system('bedtools bamtobed -i {TbamFile} -cigar | bzip > {TbedFile} && tabix {TbedFile}'.format(TbamFile=args.Tumorbam, TbedFile=TbedFile))
        os.system('bedtools bamtobed -i {NbamFile} -cigar | bzip > {NbedFile} && tabix {NbedFile}'.format(NbamFile=args.Normalbam, TbedFile=NbedFile))
    # get sqlite3 database for each bed.gz file 
    dbFile_tumor = makeupDB(TbedFile, args.TSampleID)
    dbFile_normal = makeupDB(NbedFile, args.NSampleID)
    # Load COV and MapQ 
    bg_df_T = background(args.genomeWindow, TbedFile, dbFile_tumor, showchromSpan=False, workthread=int(args.thread))
    bg_df_N = background(args.genomeWindow, NbedFile, dbFile_normal, showchromSpan=False, workthread=int(args.thread))
    SV_df_T = background(args.rawBedFile, TbedFile, dbFile_tumor, showchromSpan=True, workthread=int(args.thread))
    SV_df_N = background(args.rawBedFile, NbedFile, dbFile_normal, showchromSpan=True, workthread=int(args.thread))
    # Normalization 
    bg_df_T_nafree = bg_df_T.dropna()
    bg_df_N_nafree = bg_df_N.dropna()
    SV_df_T['COV_Zscore'] = SV_df_T['COV'].apply(lambda x: (x-np.mean(bg_df_T_nafree['COV'])) / np.std(bg_df_T_nafree['COV']))
    SV_df_T['mapQ_Zscore'] = SV_df_T['mapQRate'].apply(lambda x: (x-np.mean(bg_df_T_nafree['mapQRate'])) / np.std(bg_df_T_nafree['mapQRate']))
    SV_df_N['COV_Zscore'] = SV_df_N['COV'].apply(lambda x: (x-np.mean(bg_df_N_nafree['COV'])) / np.std(bg_df_N_nafree['COV']))
    SV_df_N['mapQ_Zscore'] = SV_df_N['mapQRate'].apply(lambda x: (x-np.mean(bg_df_N_nafree['mapQRate'])) / np.std(bg_df_N_nafree['mapQRate']))
    SV_df_T = SV_df_T.drop_duplicates()
    SV_df_N = SV_df_N.drop_duplicates()
    SV_df_T.index = SV_df_T['window']
    SV_df_N.index = SV_df_N['window']
    # Get Mis Score 
    SeqCompareDf_Filter = PairwiseCompare.MisScorePipe(args.rawBedFile).drop_duplicates()
    SeqCompareDf_Filter.index = SeqCompareDf_Filter['chrom'] + "_" + SeqCompareDf_Filter['start'].apply(str) + "-" + SeqCompareDf_Filter['end'].apply(str)
    SeqCompareDf_Filter['ABSMisScore'] = SeqCompareDf_Filter['MisScore'].apply(lambda x: np.abs(x))
    SeqCompareOut = os.path.join(args.savedir, '%s.Somatic.bed' % args.TSampleID)
    SeqCompareDf_Filter.to_csv(SeqCompareOut)
    df_SVwindow = pd.read_csv(args.rawBedFile, header=None)
    df_SVwindow.columns = ['chrom', 'start','end', 'SomSeq','SomReads','SomCount','GermSeq','GermReads','GermCount','Label']
    df_SVwindow_Filter = df_SVwindow.loc[df_SVwindow['Label']=='NormalOutput|EMOutput'].drop_duplicates()
    df_SVwindow_Filter['window'] = df_SVwindow_Filter['chrom'] + "_" +df_SVwindow_Filter['start'].apply(str) +"_"+df_SVwindow_Filter['end'].apply(str)
    df_SVwindow_Filter.index = df_SVwindow_Filter['window']
    # Merge to get all Features for each candidate SV windows 
    SVwindowList = np.intersect1d(SeqCompareDf_Filter.index, df_SVwindow_Filter)
    # Calculate adapted reads percentage from Normal and tumor 
    ReadPool = pd.concat([SV_df_T.loc[SVwindowList, ['window', 'COV_Zscore', 'mapQ_Zscore', 'chromSpan']],
                          SV_df_N.loc[SVwindowList, ['COV_Zscore', 'mapQ_Zscore', 'chromSpan']],
                          df_SVwindow_Filter.loc[SVwindowList, 'SomReads'], 
                          df_SVwindow_Filter.loc[SVwindowList, 'SomReads'].apply(lambda x: [a.split("|")[-1] for a in ",".join(x.split(";")).split(",")]) + df_SVwindow_Filter.loc[SVwindowList,'GermReads'].apply(lambda x: [a.split("|")[-1] for a in ",".join(x.split(";")).split(",")]),
                          SV_df_T.loc[SVwindowList, 'TotalReadID'].apply(lambda x: x.split(",")),
                          SV_df_T.loc[SVwindowList, 'mapQRate'],
                          SV_df_N.loc[SVwindowList, 'TotalReadID'].apply(lambda x: x.split(",")),
                          SV_df_N.loc[SVwindowList, 'mapQRate']], axis=1)
    ReadPool.columns = ['window', 'COV_Tumor', 'mapQ_Tumor', 'chromSpan_Tumor', 'COV_Normal', 'mapQ_Normal', 'chromSpan_Normal', 'SomReads', 'AdaptReads', 'TotalRead_T', 'mapQRate_T', 'TotalRead_N', 'mapQRate_N']
    ReadPool['AdaptRatio_T'] = ReadPool.apply(lambda x: np.intersect1d(x['AdaptReads'], x['TotalRead_T']).shape[0] / (len(x['TotalRead_T'])*(1-x['mapQRate_T'])) if len(x['TotalRead_T'])*(1-x['mapQRate_T'])>0 else 0, axis=1)
    ReadPool['AdaptRatio_N'] = ReadPool.apply(lambda x: np.intersect1d(x['AdaptReads'], x['TotalRead_N']).shape[0] / (len(x['TotalRead_N'])*(1-x['mapQRate_N'])) if len(x['TotalRead_N'])*(1-x['mapQRate_N'])>0 else 0, axis=1)
    ReadPool['SupportReadSpanRatio'] = ReadPool.loc[ReadPool.index, 'SomReads'].apply(lambda x: spanchrRatio([a.split("|")[-1] for a in ",".join(x.split(";")).split(",")], dbFile_tumor))
    # Load random forest model to get predicition 
    model = load('RandomForest.1218.WholeData8-2.FinalModel.joblib')
    X_scaled = ReadPool[['COV_Tumor', 'mapQ_Tumor', 'COV_Normal', 'mapQ_Normal', 'ABSMisScore', 'chromSpan_Tumor', 'chromSpan_Normal', 'AdaptRatio_T', 'AdaptRatio_N', 'SupportReadSpanRatio']]
    yprob = model.predict_proba(X_scaled)[:, 1]
    y_hat = model.predict(X_scaled)
    ReadPool['yprob'] = yprob
    ReadPool['y_hat'] = y_hat
    ReadPoolOut = os.path.join(args.savedir, 'RandomForestResult.tsv')
    ReadPool.to_csv(ReadPoolOut, sep="\t")
    return(ReadPoolOut, SeqCompareOut)

def callsomaticSV(args):
    # Handel the logic for the callsomaticSV module 
    args.rawBedFile = localGraph(args)
    ReadPoolOut, SeqCompareOut = AlnFeature(args)
    out_vcf = os.path.join(args.savedir, '%s.vcf' % args.TSampleID)
    bed2vcf(args.rawBedFile, SeqCompareOut, ReadPoolOut,out_vcf,args.TSampleID, args.Reference)

def main():
    # Main parser 
    parser = argparse.ArgumentParser(
        prog='TDScope.py',
        description='TDScope: A computational system for somatic SV calling with local graph genome optimization and whole genome alignment feature adjustment',
        epilog='''
        Use python TDScope.py <command> -h to view help information for a specific command
        Available commands:
            -localGraph: Run Local graph optimization module on candidate somatic SV window 
            -AlnFeature: collect feature from spanned reads of each potential somatic SV selected by localGrpah 
            -callsomaticSV: Call somatic SV and output in vcf format
        '''
    )
    # Create sub-command parsers
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands:',
        description='These are the commands supported by TDScope.py:'
    )
    # Add LocalGraph sub-command 
    parser_localGraph = subparsers.add_parser(
        'localGraph',
        help='Run Local graph optimization module on candidate somatic SV window',
        description='This command is used to perform the localGraph module'
    ) 
    parser_localGraph.add_argument("-w", "--windowBed", required=True, help="pre made tandem repeat windows chrom,repeatStart,repeatEnd")
    parser_localGraph.add_argument("-T", "--Tumorbam", required=True, help="ONT read alignment bam file, multiple bam should seperated with ','")
    parser_localGraph.add_argument("-N", "--Normalbam", required=True, help="ONT read alignment bam file, multiple bam should seperated with ','")
    parser_localGraph.add_argument("-t", "--TSampleID", required=True, help="SampleID for this somatic project, if multiple sample should seperated with ',' and have same length with tumor bam")
    parser_localGraph.add_argument("-n", "--NSampleID", required=True, help="SampleID for this somatic project, if multiple sample should seperated with ',' and have same length with normal bam")
    parser_localGraph.add_argument("-r", "--Reference", required=True, help="reference file fasta path")
    parser_localGraph.add_argument("-s", "--savedir", required=True, help="dir for result file save")
    parser_localGraph.add_argument("-p", "--thread", required=True, help="CPU use for program")
    parser_localGraph.add_argument("-o", "--offset", type=int, default=50, help="offset default value is 50")
    parser_localGraph.add_argument("-q", "--mapQ", type=int, default=5, help="mapQ default value is 5")
    parser_localGraph.set_defaults(func=localGraph)
    # Add AlnFeature sub-command 
    parser_AlnFeature = subparsers.add_parser(
        'AlnFeature',
        help='Run alignment feature collection and random forest process for candidate somatic SV window',
        description='This command is used to perform the AlnFeature module'
    ) 
    parser_AlnFeature.add_argument('-B', '--rawBedFile', required=True, help="path to RawBedFile from LocalGraph Step")
    parser_AlnFeature.add_argument('-W', '--genomeWindow', required=True, help="gernomic window file, by default 10kb window bed could fetch by using bedtools makewindows command")
    parser_AlnFeature.add_argument("-T", "--Tumorbam", required=True, help="ONT read alignment bam file, multiple bam should seperated with ','")
    parser_AlnFeature.add_argument("-N", "--Normalbam", required=True, help="ONT read alignment bam file, multiple bam should seperated with ','")
    parser_AlnFeature.add_argument("-t", "--TSampleID", required=True, help="SampleID for this somatic project, if multiple sample should seperated with ',' and have same length with tumor bam")
    parser_AlnFeature.add_argument("-n", "--NSampleID", required=True, help="SampleID for this somatic project, if multiple sample should seperated with ',' and have same length with normal bam")
    parser_AlnFeature.add_argument("-r", "--Reference", required=True, help="reference file fasta path must have fai file in the same pathway")
    parser_AlnFeature.add_argument("-s", "--savedir", required=True, help="dir for result file save the bed dir ")
    parser_AlnFeature.add_argument("-p", "--thread", required=True, help="CPU use for program")
    parser_AlnFeature.set_defaults(func=AlnFeature)
    # Add callsomaticSV process, a merge of localGraph and AlnFeature
    parser_callsomaticSV = subparsers.add_parser(
        'callsomaticSV',
        help='Run Full process of TDscope finally output the vcf format file'
    )
    parser_callsomaticSV.add_argument('-W', '--genomeWindow', required=True, help="gernomic window file, by default 10kb window bed could fetch by using bedtools makewindows command")
    parser_callsomaticSV.add_argument("-w", "--windowBed", required=True, help="pre made candidate somatic SV windows chrom,repeatStart,repeatEnd")
    parser_callsomaticSV.add_argument("-T", "--Tumorbam", required=True, help="ONT read alignment bam file, multiple bam should seperated with ','")
    parser_callsomaticSV.add_argument("-N", "--Normalbam", required=True, help="ONT read alignment bam file, multiple bam should seperated with ','")
    parser_callsomaticSV.add_argument("-t", "--TSampleID", required=True, help="SampleID for this somatic project, if multiple sample should seperated with ',' and have same length with tumor bam")
    parser_callsomaticSV.add_argument("-n", "--NSampleID", required=True, help="SampleID for this somatic project, if multiple sample should seperated with ',' and have same length with normal bam")
    parser_callsomaticSV.add_argument("-r", "--Reference", required=True, help="reference file fasta path")
    parser_callsomaticSV.add_argument("-s", "--savedir", required=True, help="dir for result file save")
    parser_callsomaticSV.add_argument("-p", "--thread", required=True, help="CPU use for program")
    parser_callsomaticSV.add_argument("-o", "--offset", type=int, default=50, help="offset default value is 50")
    parser_callsomaticSV.add_argument("-q", "--mapQ", type=int, default=5, help="mapQ default value is 5")
    parser_callsomaticSV.set_defaults(func=callsomaticSV)
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
