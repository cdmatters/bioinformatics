from Bio import SeqIO
import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser, FastaIterator

np.random.seed(0)

FILES = ["cyto.fasta", "mito.fasta", "nucleus.fasta", "secreted.fasta"]
def load_files():
    '''Load all files in to an arrary, unshuffled'''
    data = []
    for i, filename in enumerate(FILES):
        with open("data/"+filename) as f:
            filedata = [(values, i)for values in FastaIterator(f)]
            data.extend(filedata)
    return data

def load_test():
    with open("data/blind.fasta") as f:
        filedata = [(values, -1) for values in FastaIterator(f)]
    return filedata

def train_validate_split(split, data):
    '''1. Shuffle data, then split it according to fraction in split'''
    np.random.shuffle(data) # in place
    n = len(data)
    split_point = int(n * split)
    return (data[split_point:], data[:split_point])

def nfolds_cross_validate(n_folds, data, run_model, *args, **kwargs):
    '''Return the NFolds Cross Validation score from run model, given data.
    run_model should be function: run_model(train, validate, *args, *kwargs) '''
   
    np.random.shuffle(data) # in place

    results = []
    split_size = len(data) // n_folds
    for i in range(n_folds):
        valid = data[i*split_size : (i+1)*split_size]
        train = data[ :i*split_size] +  data[(i+1)*split_size:]
        r = run_model(train, valid, *args, **kwargs)
        results.append(r)

        if kwargs.get("verbose", None):
            print(r)
    return np.mean(np.array(results), axis=0)

def tuples_to_matrices(tuples):
    X,Y = list(zip(*tuples))
    return np.array(X), np.array(Y)

def matrices_to_tuples(X, Y):
   return list(zip(X,Y))

def dict_to_vec(sequences_list, filter_out=None):
    all_keys = set()
    [all_keys.update(s.keys()) for s in sequences_list]

    banned_keys = filter_out if filter_out is not None else []
    keys = sorted(list(all_keys - set(banned_keys)))

    to_vec = lambda x: [x.get(k, 0) for k in keys]
    return list(map(to_vec, sequences_list)), keys
    
