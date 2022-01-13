import sys
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import os
import numpy as np
from seqtools import SequenceTools
from collections import Counter


########################
## HARD CODED THINGS ##
#######################

# Firmly set the indexing of nucleotides and amino acids
NUC_ORDER = ['A', 'T', 'C', 'G']
AA_ORDER = [k.upper() for k in SequenceTools.protein2codon_.keys()]

NUC_IDX = {NUC_ORDER[i]: i for i in range(len(NUC_ORDER))}
AA_IDX = {AA_ORDER[i]: i for i in range(len(AA_ORDER))}

CODON_ORDER = [c.upper() for c in SequenceTools.codon2protein_.keys()]
CODON_IDX = {CODON_ORDER[i]: i for i in range(len(CODON_ORDER))}
    
################################
### Pre-processing functions ###
################################



class Library(object):
    """
    Class to hold hard-coded properties of library data. Defaults are for AAV5 libraries.
    """
    
    def __init__(self, name,
                 pre_filepath, 
                 post_filepath, 
                 primer_start_end=(5, 26),
                 linker_start_end=(26, 32),
                 insertion_start_end=(32, 53),
                 second_linker_start_end=(53, 62),
                 primer_seq='ATGGCCACCAACAACCAGAGA',
                 linker_seq_nuc='ACCGGT',
                 linker_seq_aa='TG',
                 second_linker_seq_nuc='GGCTTAAGT',
                 second_linker_seq_aa='GLS',
                 umi_start_end=(0,5)
                ):
        
        self.name = name
        self.pre_filepath = pre_filepath # path to pre-selection FASTQ library
        self.post_filepath = post_filepath # path to post-selection FASTQ library
        
        self.umi_idx = umi_start_end
        self.primer_idx = primer_start_end
        self.linker_idx = linker_start_end
        self.insertion_idx = insertion_start_end
        self.second_linker_idx = second_linker_start_end
        self.aa_linker = linker_seq_aa 
        self.second_aa_linker = second_linker_seq_aa
        self.nuc_linker = linker_seq_nuc
        self.second_nuc_linker = second_linker_seq_nuc
        self.primer_seq = primer_seq
        
        if self.primer_idx[1] != self.linker_idx[0]:
            self.contains_misc= True
        else:
            self.contains_misc= False
        
    def get_feature_position(self, feature):
        assert feature in ['umi', 'primer', 'linker', 'insertion', 'second_linker']
        if feature == 'umi':
            return self.umi_idx
        elif feature == 'primer':
            return self.primer_idx
        elif feature == 'linker':
            return self.linker_idx
        elif feature == 'insertion':
            return self.insertion_idx
        elif feature == 'second_linker':
            return self.second_linker_idx
        
    def get_pre_file_path(self):
        return self.pre_filepath
    
    def get_post_file_path(self):
        return self.post_filepath
    
    def get_primer_seq(self):
        return self.primer_seq
   

def get_pre_filepath(name):
    """""
    Returns path to raw FASTQ files for pre-selection library
    """""
    assert name in ['old_nnk', 'new_nnk', 'lib_b', 'lib_c', 'brain_nnk', 'brain_b', 'neuron_nnk', 'neuron_b', 'microglia_nnk', 'microglia_b', 'glia_nnk', 'glia_b']
    data_dir = 'raw' # path from SLURM submit file
    
    if name == 'old_nnk':
         return os.path.join(data_dir, "5_S5_L001_R1_001.fastq")
    elif name.split('_')[0] in ['brain', 'neuron', 'microglia', 'glia']:
        # We currently only have post-infection sequencing data for brain cells
        return None
    else:
        if name == 'lib_b':
            i = 1
        elif name == 'lib_c':
            i = 3
        elif name == 'new_nnk':
            i = 5
        return os.path.join(data_dir, "DSBZ00%i_S%i_L001_R1_001.fastq" % (i, i))


def get_post_filepath(name):
    """""
    Returns path to raw FASTQ files for post-selection library
    """""
    assert name in ['old_nnk', 'new_nnk', 'lib_b', 'lib_c', 'brain_nnk', 'brain_b', 'neuron_nnk', 'neuron_b', 'microglia_nnk', 'microglia_b', 'glia_nnk', 'glia_b']
    data_dir = 'raw' # path from SLURM submit file
    
    if name == 'old_nnk':
         return os.path.join(data_dir, "6_S6_L001_R1_001.fastq")
    elif 'brain' in name:
        if name == 'brain_b':
            i = 1
        elif name == 'brain_nnk':
            i = 2
        return os.path.join(data_dir, "DSBZ00%i_S%i_L002_R1_001.fastq" % (i, i+1))
    elif name.split('_')[0] in ['neuron', 'microglia', 'glia']:
        if name == 'neuron_b':
            i = 1
        elif name == 'microglia_b':
            i = 2
        elif name == 'glia_b':
            i = 3
        elif name == 'neuron_nnk':
            i = 4
        elif name == 'microglia_nnk':
            i = 5
        elif name == 'glia_nnk':
            i = 6
        return os.path.join(data_dir, 'DSBZ00%i_S%i_L002_R1_001.fastq' % (i, i + 10))
    else:
        if name == 'lib_b':
            i = 2
        elif name == 'lib_c':
            i = 4
        elif name == 'new_nnk':
            i = 6
        return os.path.join(data_dir, "DSBZ00%i_S%i_L001_R1_001.fastq" % (i, i))

    
def extract_read_feature(lib, read, feature):
    """
    Returns a specific feature of a read
    (one of 'umi', 'primer', 'linker', 'insertion')
    given a Library object, the sequence, desired feature
    and whether to adjust the position based on whether
    the linker can be found or not.
    """
    if feature == 'misc':
        start = lib.get_feature_position('primer')[1]
        end = lib.get_feature_position('linker')[0]
    elif feature == 'end':
        start = lib.get_feature_position('insertion')[1]
        end = len(read)
    else:
        start, end = lib.get_feature_position(feature)
    return read[start:end]

    
def hamming_distance(str1, str2):
    """ Calculates the hamming distance between two strings."""
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))
        
        
def calc_mismatches_in_primer(lib, read, read_is_primer=False):
    """
    Calculates the number of mismatches (i.e. the hamming distance) between
    the primer sequence in a read and the expected primer sequence for the 
    library. If the input 'read' sequence is only the primer, then use
    the kwarg read_is_primer=True
    """
    primer_seq = lib.get_primer_seq()
    if read_is_primer:
        primer_read = read
    else:
        primer_read = extract_read_feature(lib, read, 'primer', check_linker=False)
    return hamming_distance(primer_seq, primer_read)


def read_raw_reads(lib, pre_or_post):
    """
    Generator that yields sequence reads from a FASTQ file
    given a Library object and whether it is a 'pre' or 'post'
    selection file
    """
    assert pre_or_post in ['pre', 'post']
    
    if pre_or_post == 'pre':
        fastq_file = lib.get_pre_file_path()
    elif pre_or_post == 'post':
        fastq_file = lib.get_post_file_path()
    for record in SeqIO.parse(fastq_file, "fastq"):
        read = record.seq
        yield read
    
    
def read_raw_features(lib, pre_or_post, feature='insertion'):
    """
    Generator that yields a specific feature or list of features
    of sequence reads from a FASTQ file
    """
    for read in read_raw_reads(lib, pre_or_post):
        if type(feature) == str:
            yield extract_read_feature(lib, read, feature)
        else:
            yield [extract_read_feature(lib, read, f) for f in feature]

    
def build_reads_database(lib, pre_or_post, save_file, chunksize=1e6, include_end=False):
    """
    Builds a database of reads that is stored in a pandas DataFrame. Each row 
    contains the features of the read ('umi', 'primer', 'misc', 'linker', 'insertion', 'end')
    and the number of mismatches in the primer. 
    """
    chunksize = int(chunksize)
    feature_keys = ['umi', 'primer', 'misc', 'linker', 'insertion', 'second_linker']
    if not lib.contains_misc:
        feature_keys.remove('misc')
    if include_end:
        feature_keys.append('end')
    data = {feature_key: ['']*chunksize for feature_key in feature_keys}
    data['primer_mismatches'] = [-1] * chunksize
    
    i = 0
    total = 0
    first_save=True
    for read_features in read_raw_features(lib, pre_or_post, feature=feature_keys):
        feats = {feature_keys[i]: read_features[i] for i in range(len(feature_keys))}
        for fk in feature_keys:
            data[fk][i] = feats[fk]
        data['primer_mismatches'][i] = calc_mismatches_in_primer(lib, feats['primer'], read_is_primer=True)
        i += 1
        total += 1
        if i == chunksize:
            df = pd.DataFrame(data)
            if first_save:
                df.to_csv(save_file, index=False)
                first_save = False
            else:
                df.to_csv(save_file, mode='a', header=False, index=False)
            
            i = 0
            data = {feature_key: ['']*chunksize for feature_key in feature_keys}
            data['primer_mismatches'] = [-1] * chunksize
            print("Reads processed: %i" % total)
            
    df = pd.DataFrame(data)
    df = df.loc[df['primer_mismatches'] >= 0]
    df.to_csv(save_file, mode='a', header=False, index=False)
            
            
def build_counts_database(lib, pre_or_post, reads_file, save_file,
                          primer_mismatch_cutoff=2, chunksize=1e6, use_nucleotides=False, 
                          filter_linkers=False):
    """
    Buils a database of insertion sequence counts that is stored in a pandas DataFrame.
    Each row contains the amino acid insertion sequence and the number of times that
    sequence appears in the corresponding reads database. If the number of mismatches
    in the primer sequence of the read exceeds primer_mismatch_cutoff, then the 
    read is discarded.
    """
    chunksize = int(chunksize)
    cols = ["insertion", "primer_mismatches"]
    if filter_linkers:
        cols += ['linker', 'second_linker']
    chunks = pd.read_csv(reads_file, chunksize=chunksize, usecols=cols)
    counts = None
    translate = lambda s: str(Seq(str(s)).translate())
    i = 0
    for chunk in chunks:
        condition1 = chunk['primer_mismatches'] <= primer_mismatch_cutoff
        condition2 = chunk['primer_mismatches'] >= 0
        if filter_linkers:
            aa_linkers = chunk['linker'].apply(translate)
            aa_second_linkers = chunk['second_linker'].apply(translate)
            condition3 = (aa_linkers == lib.aa_linker)
            condition4 = (aa_second_linkers == lib.second_aa_linker)
            good_nuc_inserts = chunk['insertion'].loc[condition1 & condition2 & condition3 & condition4]
        else:
            good_nuc_inserts = chunk['insertion'].loc[condition1 & condition2]
        if not use_nucleotides:
            good_aa_inserts = good_nuc_inserts.apply(translate)
            chunk_cts = good_aa_inserts.value_counts()
        else:
            chunk_cts = good_nuc_inserts.value_counts()
        if counts is None:
            counts = chunk_cts
        else:
            counts = counts.add(chunk_cts, fill_value=0)
        
        save_counts = counts.reset_index()
        save_counts.columns = ['seq', 'count']
        save_counts.sort_values('count', ascending=False, inplace=True)
        save_counts.to_csv(save_file, index=False)
        i += 1
        print("Reads processed for counts: %i" % (i * chunksize))
        
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("library_name", help="name of library: 'lib_b', 'lib_c', 'old_nnk' or 'new_nnk'", type=str)
    parser.add_argument("pre_or_post", 
                        help="use 'pre' or 'post' to specify which of the libraries to process", type=str)
    parser.add_argument("-r", "--build_reads_db", help="if true, builds the reads database", action='store_true')
    parser.add_argument("-c", "--build_counts_db", 
                        help="if true, builds the counts database (requires a reads db)", action='store_true')
    parser.add_argument("-n", "--nucleotide_counts", help="if true, calculates nucleotide counts", action='store_true')
    parser.add_argument("-p", "--primer_mismatch_cutoff", default=2, help="cutoff to throw away reads if the primer does not match", type=int)
    parser.add_argument("-f", "--filter_linkers", help="if true, only use reads with correct linker sequences for counts (only works with AAV5 data)", action='store_true')
    
    
    args = parser.parse_args()
    name = args.library_name
    pre_or_post = args.pre_or_post
    build_reads = args.build_reads_db
    build_counts = args.build_counts_db
    use_nucs = args.nucleotide_counts
    filter_linkers = args.filter_linkers
    primer_mismatch_cutoff = args.primer_mismatch_cutoff
    
    pre_filepath = get_pre_filepath(name)
    post_filepath = get_post_filepath(name)
    
    # if anything else changes, like positions of primers, linker, etc.,
    # then you can input them into the constructor below
    lib_obj = Library(name, pre_filepath, post_filepath)
    
    reads_db = "reads/%s_%s_reads.csv" % (name, pre_or_post)
    if use_nucs:
        fname = "%s_%s_nuc_counts" % (name, pre_or_post)
    else:
        fname = "%s_%s_counts" % (name, pre_or_post)
        
    if filter_linkers:
        fname += "_filtered"
    counts_db = "counts/" + fname + ".csv"
        
    if build_reads:
        build_reads_database(lib_obj, pre_or_post=pre_or_post, 
                            save_file=reads_db, chunksize=1e6, include_end=False)
    
    if build_counts:
        build_counts_database(lib_obj, pre_or_post=pre_or_post, reads_file=reads_db, 
                              primer_mismatch_cutoff=primer_mismatch_cutoff,
                              save_file=counts_db, chunksize=1e6, 
                              use_nucleotides=use_nucs, filter_linkers=filter_linkers)

        

