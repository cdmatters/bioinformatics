from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from collections import Counter
class CategoricalClassifier(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, p_vector=[0.25, 0.25, 0.25, 0.25]):
        """
        Called when initializing the classifier
        """
        self.p_vector = p_vector

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        counts = np.array([v for k,v in sorted(Counter(y).items(), key=lambda x:x[0])])

        self.p_vector = counts/np.sum(counts)

        return self

    def predict(self, X, y=None):
        try:
            y_pred = np.random.choice([0,1,2,3], size=X.shape[0],replace=True, p=self.p_vector )
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return(y_pred)

    def score(self, X, y=None):
        # counts number of values bigger than mean
        return np.mean((self.predict(X) == y).astype(np.int32))


if __name__=="__main__":
    CategoricalClassifier(BaseEstimator, ClassifierMixin) 