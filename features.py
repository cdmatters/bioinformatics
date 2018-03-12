from aux import load_files, tuples_to_matrices, matrices_to_tuples, dict_to_vec

from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser, FastaIterator
from Bio.Alphabet import Alphabet
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import defaultdict, Counter

### FEATURES

def sequence_length(seq):
    return {'seq_length': len(seq)}

def letter_fractions(seq):
    letters = Counter(str(seq))
    total = sum(letters.values())
    
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXVYZ"
    return {"letter_{}".format(l): (letters[l]/total) for l in alphabet}

def prot_param_features(seq):
    features = {}

    pa = ProteinAnalysis(str(seq.seq).replace('X','').replace('B',''))

    # 1. Amino Acid Percent
    aa = pa.get_amino_acids_percent()
    aa_dict = {"frac_{}".format(k):v for k,v in aa.items()}
    features.update(aa_dict)
    
    # 2. Aromaticity
    features["aromaticity"] = pa.aromaticity()

    # 3. Isoelectric Point
    features["isoelectric"] = pa.isoelectric_point()

    # 4. Molecular Weight
    features["molecular_weight"] = pa.molecular_weight()

    # 5. Secondary Structure Fraction
    struc = ["struc_helix", "struc_turn", "struc_sheet"]
    ss = pa.secondary_structure_fraction()
    features.update(dict(zip(struc, ss)))

    return features


### MAPPING

def sequence_to_features(seq):

    features = {}

    features.update(sequence_length(seq))
    features.update(prot_param_features(seq))
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