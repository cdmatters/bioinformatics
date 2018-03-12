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

    def accuracy(self, train, test):
        train_X, train_Y = aux.tuples_to_matrices(train)
        test_X, test_Y = aux.tuples_to_matrices(test)
        return self.clf.score(train_X, train_Y), self.clf.score(test_X, test_Y)

    def run(self, train, validate):
        self.train(train)
        return self.accuracy(train, validate)