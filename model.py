from sklearn.ensemble import RandomForestClassifier

import aux

class ModelRandomForest:
    def __init__(self, *args, **kwargs):
        self.clf = RandomForestClassifier(*args, **kwargs)
    
    def train(self, train):
        X,Y = aux.tuples_to_matrices(train)
        self.clf.fit(X,Y)

    def predict(self, X):
        return self.clf.predict(X)

    def test_accuracy(self, test):
        X,Y = aux.tuples_to_matrices(test)
        return self.clf.score(X, Y)

    def run(self, train, validate):
        self.train(train)
        return self.test_accuracy(validate)