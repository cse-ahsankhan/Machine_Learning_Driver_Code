import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd 
import pandas_profiling as pp
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px 

from sklearn.impute import SimpleImputer
from impyute.imputation.cs import fast_knn
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import RFE

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor, VotingClassifier, VotingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score, explained_variance_score, mean_squared_error, mean_absolute_error, classification_report, roc_curve
from math import sqrt
from collections import Counter



def print_line():
    print("----------------------------------------------------------")

#Classification

def train_clf(train_x, test_x, train_y, test_y, clf):
    clf.fit(train_x, train_y)
    res = classification_report(test_y, clf.predict(test_x))
    return res

def CatBoostModelClassification(train_x, test_x, train_y, test_y):
    model=CatBoostClassifier(custom_loss=['Accuracy'],
                             random_seed=42,
                             logging_level='Silent'
                            )
    model.fit(train_x, train_y, eval_set=(test_x, test_y), plot=True)
    


def ClassificationModel(X, val_x, y, val_y, test_size = 0.25, shuffle = True, normalize = False):
    if normalize == True:
        si = StandardScaler()
        X = si.fit_transform(X)
    train_x, test_x, train_y, test_y = X, val_x, y, val_y
#     train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    xtc = ExtraTreeClassifier()
    rfc = RandomForestClassifier()
    xgb = XGBClassifier()
    dtc = DecisionTreeClassifier()
    mlp = MLPClassifier()
    lgbm = LGBMClassifier()
    ada = AdaBoostClassifier()
    knc = KNeighborsClassifier()
    svc = SVC()
    gnb = GaussianNB()
    
    def voting(train_x, test_x, train_y, test_y):
        clf = VotingClassifier(estimators=[
            ('xtc', xtc), 
            ('rfc', rfc), 
            ('xgb', xgb),
            ('dtc', dtc),
            ('mlp', mlp),
            ('lgbm', lgbm),
            ('ada', ada),
            ('knc', knc),
            ('svc', svc),
            ], 
            voting='hard')
        return train_clf(train_x, test_x, train_y, test_y, clf)


    print("Accuracy of Extra Tree Classifier is \n\n{}%".format(train_clf(train_x, test_x, train_y, test_y, xtc)))
    print_line()
    print("Accuracy of Decision Tree Classifier is \n\n{}%".format(train_clf(train_x, test_x, train_y, test_y, dtc)))
    print_line()
    print("Accuracy of Random Forest Classifier is \n\n{}%".format(train_clf(train_x, test_x, train_y, test_y, rfc)))
    print_line()
    print("Accuracy of XGBoost Classifier is \n\n{}%".format(train_clf(train_x, test_x, train_y, test_y, xgb)))
    print_line()
    print("Accuracy of Multi Layer Perceptron Classifier is \n\n{}%".format(train_clf(train_x, test_x, train_y, test_y, mlp)))
    print_line()
    print("Accuracy of AdaBoost Classifier is \n\n{}%".format(train_clf(train_x, test_x, train_y, test_y, ada)))
    print_line()
    print("Accuracy of K Neighbors Classifier is \n\n{}%".format(train_clf(train_x, test_x, train_y, test_y, knc)))
    print_line()
    print("Accuracy of Support Vector Classifier is \n\n{}%".format(train_clf(train_x, test_x, train_y, test_y, svc)))
    print_line()
    print("Accuracy of Gaussian Naive Bayes Classifier is \n\n{}%".format(train_clf(train_x, test_x, train_y, test_y, gnb)))
    print_line()
    print("Accuracy of LGBM Classifier is \n\n{}%".format(train_clf(train_x, test_x, train_y, test_y, lgbm)))
    print_line()
    print("Accuracy of Voting Classifier is \n\n{}%".format(voting(train_x, test_x, train_y, test_y)))
    print_line()
    CatBoostModelClassification(train_x, test_x, train_y, test_y)
    
    
    
def SelectBestFeaturesClassification(X, y, no_of_features, steps = 10):
    xgb = XGBClassifier()
    lgb = LGBMClassifier()
    rfc = RandomForestClassifier()
    
    sel1 = RFE(xgb, n_features_to_select=no_of_features, step=steps, verbose=2)
    sel2 = RFE(lgb, n_features_to_select=no_of_features, step=steps, verbose=2)
    sel3 = RFE(rfc, n_features_to_select=no_of_features, step=steps, verbose=2)
    
    sel1 = sel1.fit(X,y)
    sel2 = sel2.fit(X,y)
    sel3 = sel3.fit(X,y)
    
    features1 = list(zip(df.columns, sel1.support_))
    features2 = list(zip(df.columns, sel2.support_))
    features3 = list(zip(df.columns, sel3.support_))
    
    print("List of features to be selected and rejected by xgb are \n{}".format(features1))
    print_line()
    print("List of features to be selected and rejected by lgb are \n{}".format(features2))
    print_line()
    print("List of features to be selected and rejected by rfc are \n{}".format(features3))
    print_line()
    
    selected_features1 = []
    for i in features1:
        if i[1] == True:
            selected_features1.append(str(i[0]))
    print("List of features to be selected by xgb\n{}".format(selected_features1))
    print_line()
    
    selected_features2 = []
    for i in features2:
        if i[1] == True:
            selected_features2.append(str(i[0]))
    print("List of features to be selected by lgb \n{}".format(selected_features2))
    print_line()
    
    selected_features3 = []
    for i in features3:
        if i[1] == True:
            selected_features3.append(str(i[0]))
    print("List of features to be selected by rfc\n{}".format(selected_features3))
    print_line()
    
    selected_features = [x for x in selected_features1 if x in selected_features2 and x in selected_features3]
    
    print('Final list of selected features are {}'.format(selected_features))
    
    nX = X[selected_features]
    ny = y

    return nX, ny