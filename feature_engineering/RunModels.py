
from re import split
import sys,os
sys.path.append( os.path.join(".."))
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve,accuracy_score
from sklearn.pipeline import Pipeline
from UtilityFunctions import CommonHelpers,PreprocessHelpers,FeatureEngineering
from sklearn import metrics
from sklearn.metrics import f1_score
class Split_Data:
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y

    def get_split(self,split_ratio=0.2,random_state = 473):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, 
        self.Y, test_size=split_ratio, random_state=random_state)

    def get_x_split(self):
        return (self.x_train,self.x_test)

    def get_y_split(self):
        return (self.y_train,self.y_test)

class Run_Model:
    def __init__(self,model,split_obj:Split_Data):
        self.model = model
        if split_obj is None:
            raise Exception("Split data not available")
        self.x_train, self.x_test=split_obj.get_x_split()
        self.y_train, self.y_test = split_obj.get_y_split() 

    def run_model(self):
        self.fit_model = self.model.fit(self.x_train,self.y_train)
        self.result = self.fit_model.predict(self.x_test)

    def get_predictions(self):
        return self.result

    def get_metrics(self):   
        # self.score = metrics.f1_score(self.y_test, self.result, pos_label=list(set(self.y_test)))
        self.confusion = confusion_matrix(self.y_test, self.result)
        self.accuracy =accuracy_score(self.y_test, self.result)
        self.f1=f1_score(self.y_test, self.result, average='macro')
        return [self.confusion,self.accuracy,self.f1]
