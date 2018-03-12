from aux import load_files, tuples_to_matrices, matrices_to_tuples, dict_to_vec

from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser, FastaIterator
from Bio.Alphabet import Alphabet
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np

from collections import defaultdict, Counter

### FEATURES

def sequence_length(seq):
    return {'seq_length': len(seq)}

def early_late_fractions(seq):
    seq_str = str(seq.seq)
    start = seq_str[:20]
    last = seq_str[-20:]

    start_letters = Counter(start)
    start_total = sum(start_letters.values())
    end_letters = Counter(last)
    end_total = sum(end_letters.values())
    
    alphabet = "ACDEFGHIJKLMNOPQRSTUVWVYZ"
    features = {}
    features.update({"start_letter_{}".format(l): (start_letters[l]/start_total) for l in alphabet})
    features.update({"end_letter_{}".format(l): (end_letters[l]/end_total) for l in alphabet})
    return features

def early_late_ngrams(seq):
    seq_str = str(seq.seq)

    window = 10
    start = seq_str[:window]
    end = seq_str[-window:]

    start_trigrams = Counter([ start[i:i+2] for i in range(window-2) ])
    end_trigrams = Counter([ end[i:i+2] for i in range(window-2) ])
    
    features = {}

    features.update({"start_{}".format(t):c for t,c in start_trigrams.items() })
    features.update({"end_{}".format(t):c for t,c in end_trigrams.items() })
    return features 

def prot_param_features(seq):
    features = {}

    pa = ProteinAnalysis(str(seq.seq)) # .replace('X','G').replace('B','A')

    # 1. Amino Acid Percent
    aa = pa.get_amino_acids_percent()
    aa_dict = {"frac_{}".format(k):v for k,v in aa.items()}
    features.update(aa_dict)
    
    # 2. Aromaticity
    features["aromaticity"] = pa.aromaticity()

    # 3. Isoelectric Point
    features["isoelectric"] = pa.isoelectric_point()

    # 4. Molecular Weight
    try:
        features["molecular_weight"] = pa.molecular_weight()
    except ValueError:
        replaced = str(seq.seq).replace('X', 'G').replace('B', 'N')
        features["molecular_weight"] = ProteinAnalysis(replaced).molecular_weight()
        
    # 5. Flexibility
    # try:
    #     features["flexibility"] = np.mean(pa.flexibility())
    # except KeyError:
    #     replaced = str(seq.seq).replace('X', 'G').replace('B', 'N').replace('U','C')
    #     features["flexibility"] = np.mean(ProteinAnalysis(replaced).flexibility())

    # 6. Secondary Structure Fraction
    struc = ["struc_helix", "struc_turn", "struc_sheet"]
    ss = pa.secondary_structure_fraction()
    features.update(dict(zip(struc, ss)))

    return features


### MAPPING

def sequence_to_features(seq):

    features = {}

    features.update(sequence_length(seq))
    features.update(prot_param_features(seq))
    features.update(early_late_fractions(seq))
    # features.update(early_late_ngrams(seq))
    return features

def map_sequences_to_features(data_tuples):
    '''Transfrom list of [(Seq, Label), ...] -> [(Feature, Label)...] '''
    X, Y = tuples_to_matrices(data_tuples)
    X_features = list(map(sequence_to_features, X))
    X_feature_vec, keys = dict_to_vec(X_features, filter_out=None)
    print(keys)
    return matrices_to_tuples(X_feature_vec, Y)

if __name__=="__main__":

    import random
    two_proteins = load_files()[0:2]
 
    print(map_sequences_to_features(two_proteins))