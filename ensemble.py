import pickle
from sklearn import tree
import numpy as np 
# from feature import NPDFeature
# from ensemble import AdaBoostClassifier
import numpy as np 
from pylab import *
from PIL import Image
import math
# import math
from sklearn import metrics
from sklearn import tree
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier
        
        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.DTC = weak_classifier
        self.num = n_weakers_limit

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        w = np.ones(X.shape[0])/X.shape[0]#初始化权重
        # clf1 = clf.fit(X,y)
        # preds = clf.predict(X)
        model_w = []
        model = []
        for n in range(0,10):
            clf = tree.DecisionTreeClassifier(max_depth=50,min_samples_leaf=50,random_state=30,criterion='gini')
            clf = clf.fit(X,y,w)
            preds = clf.predict(X)
            num = 0
            e = 0
            model.append(clf)
            for z in zip(y,preds):
                if z[0] != z[1]:
                    e = e + w[num]
                num = num + 1
        #     e = 1 - clf.score(X,y,w)
            print(e)
            gramma = 0.5*math.log((1-e)/e)
            model_w.append(gramma)
            Z = 0
            for i in range(0,150):
                Z = Z + w[i]*math.exp(-gramma*preds[i]*y[i])
            print(Z)
            for i in range(0,150):
                w[i] = (w[i]*math.exp(-gramma*y[i]*preds[i]))/Z
            print(w)
            print('------------------------------------------------')
        self.save(model, 'model.txt')
        self.save(model_w, 'model_w.txt')
        
    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        model = load('model.txt')
        model_w = load('model_w.txt')
        y_preds = []
        num = 0
        for m,w in zip(model,model_w):
            num = num + 1
            if num == 1:
                y_preds = w*(m.predict(X))
            else :
                y_preds = y_preds + w*(m.predict(X))
            
        return y_preds

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
