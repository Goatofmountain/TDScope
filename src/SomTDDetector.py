''''
__Author__: Kailing Tu
__Version__: v18.1.0
__ReleaseTime__: 2024-11-26

Release Note:
    Change default parameter mapQ = 5, offset=200
    Remove TD Number and time cutoff for program stop
    Merge pipeline DataMaker and DecisionMaker together for faster work 
    Using Cython script to accelerate datamaker and EM speed 

Requirement:
    os,re, argparse
    DecisionMaker : From our script 
'''
import os,re
import argparse
from DecisionMaker import *
from DataScanner import *
import functools
from multiprocessing import Pool
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def TDscope(TDRecord, DataMaker, DataMaker2, DecisionMaker):
    '''
    Complete pipeline of TDscope 
    :param:
        TDRecord: bed record for Tandem Repeat window with \t split 
        DataMarker: data extractor from bam Files, preset by functools in the main pipeline 
        DecisionMaker: pipeline decide whether a TD region is somatic or not
    :return:
        Record: TD region record including chrom,start,end, somTDconsensus seq, somTD support reads, germTDconsensus seq, germTD support reads, pvalue
    '''
    start_time = time.time()
    logging.info(f"pipeline for region {TDRecord} start to work")
    sequenceList, ReadIDs, flank_5,flank_3, TDRecord, flag= DataMaker(TDRecord)
    SVType = TDRecord.strip().split("\t")[3]
    Record = DecisionMaker(TDRecord, sequenceList, ReadIDs, flank_5, flank_3, flag)
    if (Record[-1].split("|")[-1] != 'EMOutput') and (SVType=='DUP'):
        logging.info(f"region {TDRecord} gonna be checked another time for potential unspanned Duplication status")
        reSCANDat = DataMaker2(TDRecord)
        sequenceList_5, ReadIDs_5, flank_5_5,flank_3_5, TDRecord, flag5 = reSCANDat[0]
        sequenceList_3, ReadIDs_3, flank_5_3,flank_3_3, TDRecord, flag3 = reSCANDat[1]
        Record5 = DecisionMaker(TDRecord, sequenceList_5, ReadIDs_5, flank_5_5, flank_3_5, flag5)
        if Record5[-1].split("|")[-1] == 'EMOutput':
            Record = Record5
        else:
            Record3 = DecisionMaker(TDRecord, sequenceList_3, ReadIDs_3, flank_5_3, flank_3_3, flag3)
            if Record3[-1].split("|")[-1] == 'EMOutput':
                Record = Record3
            else:
                if (len([x for x in np.setdiff1d(ReadIDs_5, ReadIDs) if re.search("_tumor",x)]) >=3):
                    Record[-1] = flag5
                elif (len([x for x in np.setdiff1d(ReadIDs_3, ReadIDs) if re.search("_tumor",x)]) >=3):
                    Record[-1] = flag3
    time_span = time.time() - start_time
    logging.info(f"pipeline for region {TDRecord} finished Take {time_span}s")
    return(Record)

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
    P = Pool(processes=int(args.thread))
    results = []
    for TDRecord in TDRecordList:
        result = P.apply_async(TDscope, (TDRecord,DataMaker_exe,DataMaker2_exe,Decision_exe,))
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
    parser.add_argument("-o", "--offset", type=int, default=200, help="offset default value is 200")
    parser.add_argument("-q", "--mapQ", type=int, default=5, help="mapQ default value is 5")
    args = parser.parse_args()
    main(args)


