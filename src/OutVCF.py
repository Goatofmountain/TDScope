import os,re
import argparse
import time
import pandas as pd
import numpy as np

def parse_fasta(fai):
    """Parse a genome into a tuple of bases. """
    chromosomes = {}
    with open(fai) as infile:
        for line in infile:
            chrom = line.split('\t')[0]
            length = line.split('\t')[1]
            chromosomes[chrom] = length
    return(chromosomes)

def generate_vcfheader(chromosomes,out_vcf,fasta):
    Info ='''##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length of the SV">\n##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the SV">\n##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description="Number of reads supporting the structural variation">\n##INFO=<ID=RNAMES,Number=.,Type=String,Description="Names of supporting reads">\n##INFO=<ID=AF,Number=1,Type=Float,Description="Allele Frequency">\n'''
    Tools = '''##fileformat=VCFv4.2\n##source=TDscope.1.0\n##FILTER=<ID=PASS,Description="All filters passed">\n'''
    with open(out_vcf,'w') as vcf:
        ### Tools 
        vcf.write(Tools)
        ### Data
        current_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
        vcf.write('''##fileDate="'''+current_time+'''"\n''')
        ### reference info
        vcf.write('''##reference='''+fasta+'\n')
        for chrom,length in chromosomes.items():
            vcf.write('''##contig=<ID='''+chrom+',length='+length+'>\n')
        ### SV info
        vcf.write('''##ALT=<ID=INS,Description="Insertion">\n##ALT=<ID=DEL,Description="Deletion">\n''')
        ### format
        vcf.write('''##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n''')
        ### INFO
        vcf.write(Info)
    return(out_vcf)

def bed2vcf(input_bed1, input_bed2, input_bed3, out_vcf,TumorID, reference):
    df_Raw = pd.read_csv(input_bed1, sep="\t", header=None).drop_duplicates()
    df_Raw['window'] = df_Raw[0]+"_"+df_Raw[1].apply(str)+"-"+df_Raw[2].apply(str)
    df_Raw.index = df_Raw['window']
    df_Som = pd.read_csv(input_bed2, sep="\t", header=None).drop_duplicates()
    df_Som.index = df_Som[3]
    df_model = pd.read_csv(input_bed3, sep="\t", index_col=0)
    chromosomes = parse_fasta('%s.fai' % reference)
    generate_vcfheader(chromosomes, out_vcf, reference)
    with open(out_vcf,'a') as vcf:
        vcf.write(("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}\n").format(TumorID))
        for i in df_model.index:
            line_Raw = list(df_Raw.loc[i])
            line_Som = list(df_Som.loc[i])  
            chrom,start,end = line_Raw[0], str(line_Raw[1]),line_Raw[2]
            window = i
            supportReads = line_Som[4].split(";")[0]
            somaticSeq = ",".join(line_Raw[3].split(";"))
            germlinSeq = ",".join(line_Raw[6].split(";"))
            SVLen = int(line_Som[-3])
            AF = line_Som[-2]
            yprob = df_model.loc[i, 'yprob']
            yhat = df_model.loc[i, 'y_hat']
            SVType = 'MisAlign'
            if SVLen >= 50:
                    SVType = 'INS'
            elif SVLen <= -50:
                SVType ='DEL'
            SVID = 'TDscope.'+SVType+'.'+window
            Ref = germlinSeq
            QUAL = '.'
            FILTER = 'PASS'
            FORMAT='GT'
            INFO = 'SVLEN={svlen};SVTYPE={SVType};END={end};SUPPORT={supportnum};RNAMES={supportReads};AF={AF};ConfidenceSV={yprob};DecisionSV={yhat}'.format(
                svlen=SVLen,SVType=SVType,end=end,supportnum=len(supportReads.split(',')),
                supportReads=supportReads,AF=AF, yprob=yprob, yhat=yhat)
            vcf.write('\t'.join([chrom,start,SVID,Ref,somaticSeq,QUAL,FILTER,INFO,FORMAT,'0/1\n']))
    return(out_vcf)

def main(args):
    input_bed1 = os.path.join(args.rawdir, '{sampleID}/{sampleID}.vs.{sampleID}.TandemRepeat.Raw.bed'.format(sampleID=args.sampleID))
    input_bed2 = os.path.join(args.validatedDir, '{sampleID}.Somatic.bed'.format(sampleID=args.sampleID))
    input_bed3 = args.modelpredicion
    out_vcf = os.path.join(args.outputDir, "{sampleID}.MisScore50.vcf".format(sampleID=args.sampleID))
    bed2vcf(input_bed1, input_bed2, input_bed3, out_vcf, args.sampleID, args.reference)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-R", "--reference", required=True, help="reference path should contain fai file at the same dir")
    parser.add_argument("-r", "--rawdir", required=True, help="raw Dir")
    parser.add_argument("-v", "--validatedDir", required=True, help="som Dir for .somatic.bed by Step3")
    parser.add_argument('-m', "--modelprediction", required=True, help='predicition file for candidate somatic SVs')
    parser.add_argument("-s", "--sampleID", required=True, help="sampleID")
    parser.add_argument("-o", "--outputDir", required=True, help="output dir")
    args = parser.parse_args()
    main(args)



