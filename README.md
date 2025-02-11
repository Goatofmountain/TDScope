# TDScope
## Introduction
A somatic structural variation caller based on long-read technology
The TDScope algorithm utilizes full-length read sequence information within the candidate somatic SV interval. It optimizes local graph genomes through sPOA graph and sequence mixture models to detect somatic SV events. This strategy effectively reduces the impact of alignment errors in long-read sequences and demonstrates superior performance in resolving low-complexity sequences (including low-complexity and tandem repeat sequences) and highly heterogeneous local genomic information.

## Features
- Utilizes full-length sequence from long-read data
- Optimizes local graph genome with sequence mixture model 
- Provides detailed sequence and support read ID for each componenet of local genome including somatic conponents

## Installation
To get started with this software, follow these steps:

### Dependencies
Ensure you have the following Python packages installed:
- pysam version 0.19.1
- pyspoa version 0.2.1
- numpy version 1.21.5
- scipy version 1.7.3
- sklearn version 1.0.2
- Levenshtein version 0.23.0

### Installation Steps
- Clone the repository:
   ```bash
   git clone https://github.com/Goatofmountain/TDScope.git
   cd TDScope
   ```

## Usage
The TDScope algorithm consists of two main modules: the local graph genome optimization module (`localGraph`), which optimizes the local graph genome, and the local graph confidence assessment module (`AlnFeature`), which evaluates the confidence of the local graph genome. We have designed the `callsomaticSV` module to link the above modules together to directly obtain the somatic SV calculation results in VCF format. The specific usage is as follows:
### command line
```bash
python src/TDScope.py callsomaticSV \
    -W <Genome interval window> \             # genomic window for normalization by default 10kb window as we provided in doc/hg38_mainChr.10kb.window.bed.
    -w <Candidate SVwindows> \                # Candidate genomic interval TDScope build local graph genome for.
    -T <CaseBam> \                            # Path of Case sample long-read data alignment data in bam format, we recommand to use minimap2.22 for reads alignment.
    -N <ControlBam> \                         # Path of Control sample long-read data alignment data in bam format, we recommand to use minimap2.22 for reads alignment.
    -t <CaseID> \                             # Case SampleID 
    -n <ControlID> \                          # Control SampleID
    -r <Reference sequence> \                 # reference file in fasta format 
    -s <SaveDir> \                            # path for result output
    -p <Thread> \                             # Number of CPU used for calculation
    -o <offset> \                             # Length of flank region around target somatic SV region, by default 50
    -q <MapQ> \                               # Mapping quality cutoff for reads alignment record adapted for calculation, by default 5
```
### output
- \<CaseID\>.vs.\<ControlID\>.Raw.bed             # Local genome component phasing result of TDScope in bed format, consist of 10 columns
- \<CaseID\>.vcf                                # result of somatic SV calling.
#### \<CaseID\>.vs.\<ControlID\>.Raw.bed Description
| Column | Name | Description |
|--------|------|-------------|
| 1      | Chromosome | Chromosome name |
| 2      | Start | Start position of the region |
| 3      | End | End position of the region |
| 4      | somatic Sequence | sequence of somatic genome, split by ";" if more than 1 components |
| 5      | support reads for somatic component | ID of reads supporting somatic genome components, split by ";" if more than 1 components |
| 6      | Number of somatic component | Number of somatic component |
| 7      | germline Sequence | sequence of germline genome, split by ";" if more than 1 components |
| 8      | support reads for germline component | ID of reads supporting germline genome components, split by ";" if more than 1 components |
| 9      | Number of germline component | Number of germline component |
| 10     | label | Label of interval |
