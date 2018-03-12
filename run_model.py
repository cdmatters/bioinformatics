from features import map_sequences_to_features
from model import ModelRandomForest 
from aux import load_files, train_validate_split, nfolds_cross_validate, tuples_to_matrices

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import numpy as np

def grid_search():
    file_data = load_files()
    feature_data = map_sequences_to_features(file_data)
    X, Y = tuples_to_matrices(feature_data)

    parameters = { 
        "n_estimators": [200, 300, 400, 450],
        "criterion": ["gini"],
        "n_jobs": [-1]
    }

    estimator = RandomForestClassifier()
    clf = GridSearchCV(estimator, parameters, cv=5)
    clf.fit(X,Y)
    print(clf.best_score_)
    print(clf.best_params_)
    print(clf.cv_results_["mean_test_score"])
    print(clf.cv_results_["params"])

    
def main():
    tv_split = 0.2

    file_data = load_files()
    feature_data = map_sequences_to_features(file_data)
    
    np.random.shuffle(feature_data)

    train, validate = train_validate_split(tv_split, feature_data)
    
    parameters = { 
        "n_estimators": 400,
        "criterion": "gini",
        "n_jobs": -1
    }


    X, Y = tuples_to_matrices(feature_data)
    model = RandomForestClassifier(**parameters)
    
    def run(train, validate):
        model.fit(*tuples_to_matrices(train))
        return model.score(*tuples_to_matrices(validate))

    valid_acc = nfolds_cross_validate(5, feature_data, run)
    print(valid_acc)
    

if __name__=="__main__":
    main()
    # grid_search()