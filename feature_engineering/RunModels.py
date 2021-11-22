
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

class Run_Model:
    def __init__(self,model,all_vectors,all_labels):
        self.model = model
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(all_vectors, 
        all_labels, test_size=0.20, random_state=4)
        


    def run_model(self):
        self.fit_model = self.model.fit(self.x_train,self.y_train)
        self.result = self.fit_model.predict(self.x_test)

    def get_predictions(self):
        return self.result

    def get_metrics(self):   
        # self.score = metrics.f1_score(self.y_test, self.result, pos_label=list(set(self.y_test)))
        self.confusion = confusion_matrix(self.y_test, self.result)
        self.accuracy =accuracy_score(self.y_test, self.result)
        
        return (self.confusion,self.accuracy)

