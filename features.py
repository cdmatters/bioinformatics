from aux import load_files, tuples_to_matrices, matrices_to_tuples

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
    # print(str(seq.seq))
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
    X_feature_vec = dict_to_vec(X_features)
    
    return matrices_to_tuples(X_feature_vec, Y)
    
### AUX

def dict_to_vec(sequences):
    banned_keys = ["letter_B"]
    keys = sorted(list(set(sequences[0].keys()) - set(banned_keys)))

    to_vec = lambda x: [x[k] for k in keys]
    print(keys)
    return list(map(to_vec, sequences))

if __name__=="__main__":

    import random
    two_proteins = load_files()[0:2]
 
    print(map_sequences_to_features(two_proteins))