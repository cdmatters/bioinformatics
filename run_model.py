from features import map_sequences_to_features
from aux import load_files, train_validate_split, nfolds_cross_validate, tuples_to_matrices
from categorical import CategoricalClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

import numpy as np
from time import time

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



# def get_models_and_params():
#     return [
#     (AdaBoostClassifier(),{ 
#         "n_estimators": [50, 100, 200, 400, 450, 500],
#         "learning_rate": [0.5, 1, 2],
#     }),
#     (RandomForestClassifier(),{ 
#         "n_estimators": [200, 300, 400, 450, 500],
#         "criterion": ["gini"],
#         "n_jobs": [-1]
#     }),    
#     (OneVsRestClassifier(RandomForestClassifier()),{ 
#         "estimator__n_estimators": [200, 300, 400, 450],
#         "estimator__criterion": ["gini"],
#         "estimator__n_jobs": [-1]
#     }),
#     ]

def get_models_and_params():
    return [
    (CategoricalClassifier(),{ 
        "p_vector": [[0.25, 0.25, 0.25, 0.25]]
    })
    ]


def run_test_models():
    tv_split = 0.2
    file_data = load_files()
    feature_data, keys = map_sequences_to_features(file_data)
    np.random.shuffle(feature_data)
    
    train_and_validate, test = train_validate_split(tv_split, feature_data)
    X, Y = tuples_to_matrices(train_and_validate)

    model_and_grid = get_models_and_params()
    fitted_models = []
    for model, grid in model_and_grid:
        print(model)
        clf = GridSearchCV(model, grid, cv=5)
        clf.fit(X,Y)
        fitted_models.append(clf)
    return fitted_models, train_and_validate, test
    

def get_results(classifier_lists, test):
    X, Y = tuples_to_matrices(test)
    for i, clf in enumerate(classifier_lists):
        print("Model", i)
        print("Score, ",clf.best_score_)
        print("Params", clf.best_params_)
        print("Mean Test Score ", clf.cv_results_["mean_test_score"])
        print("F1: " )
        print("AUC: " )

    
    
def main():
    tv_split = 0.2


    file_data = load_files()
    feature_data, keys = map_sequences_to_features(file_data)
    np.random.shuffle(feature_data)

    train, validate = train_validate_split(tv_split, feature_data)
    
    # parameters = { 
    #     "n_estimators": 400,
    #     "criterion": "gini",
    #     "n_jobs": -1
    # }


    # X, Y = tuples_to_matrices(feature_data)
    # model = RandomForestClassifier(**parameters)
    
    model = CategoricalClassifier()
    X,Y = tuples_to_matrices(train)

    from collections import Counter
    counts = [v for k,v in sorted(Counter(Y).items(), key=lambda x:x[0])]
    print(counts/np.sum(counts))
    model.fit(X,Y)
    print(model.p_vector)
    X_v, Y_v = tuples_to_matrices(validate)
    counts_v = np.array([v for k,v in sorted(Counter(Y_v).items(), key=lambda x:x[0])])
    print(counts_v/np.sum(counts_v))

    print(model.score(X_v, Y_v))
   
    Y_pred = model.predict(X_v)
    counts_v = np.array([v for k,v in sorted(Counter(Y_pred).items(), key=lambda x:x[0])])
    print(counts_v/np.sum(counts_v))



    def run(train, validate):
        model.fit(*tuples_to_matrices(train))
        return model.score(*tuples_to_matrices(validate))

    valid_acc = nfolds_cross_validate(5, feature_data, run)
    print(valid_acc)
    

if __name__=="__main__":
    # run_test_models()
    main()
    # grid_search()