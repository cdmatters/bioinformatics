from features import map_sequences_to_features
from model import ModelRandomForest 
from aux import load_files, train_validate_split, nfolds_cross_validate

import random 
random.seed(0)


def main():
    tv_split = 0.2

    file_data = load_files()
    feature_data = map_sequences_to_features(file_data)
    

    train, validate = train_validate_split(tv_split, feature_data)
    
    model = ModelRandomForest()
    print(model.run(train, validate))

    train_acc, valid_acc = nfolds_cross_validate(5, feature_data, model.run)
    print(valid_acc)
    

if __name__=="__main__":
    main()