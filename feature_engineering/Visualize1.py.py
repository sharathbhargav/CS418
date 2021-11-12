from gensim import models
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader
import sys,os
sys.path.append( os.path.join(".."))
from UtilityFunctions import CommonHelpers,PreprocessHelpers,FeatureEngineering
import tsne
import pandas as pd
import numpy as np
import seaborn as sns

book_pickle_path="../data/preprocessed/hundred_books_preprocessed.pickle"
name_pickle_path = "../data/preprocessed/hundred_names.pickle"
class Visualize:
    def init_data(self,book_pickle_path,name_pickle_path):
        self.RAW_BOOK_FOLDER = os.path.join("..","data_collection","raw_books")
        self.PREPROCESSED_STORE_LOC = os.path.join("..","data","preprocessed")
        self.google_corpus_path  = "../models/GoogleNews-vectors-negative300.bin"
        self.model_google = KeyedVectors.load_word2vec_format(self.google_corpus_path,binary=True)

        self.list_of_words = CommonHelpers.load_pickle(book_pickle_path)
        self.book_names =  CommonHelpers.load_pickle(name_pickle_path)

    def load_data(self):

        train_words = self.list_of_words[:90]
        train_names = self.book_names[:90]
        test_words  = self.list_of_words[90:]
        test_names = self.book_names[90:]
        self.train_vectors = FeatureEngineering.get_avg_feature_vectors(train_words,self.model_google,300)
        self.test_vectors = FeatureEngineering.get_avg_feature_vectors(test_words,self.model_google,300)


def get_number_id(file_name):
    import re 
    regex = r"([a-z_A-Z]+)(\d+)"
    match = re.search(regex, file_name)
    if match!=None:
        return match.group(2)


df = pd.read_csv("../data/pg_catalog.csv")
df.head(50)
df["category"] = df["LoCC"].str[:1]
category_list_train = [int(get_number_id(each)) for each in train_names]
category_list_test = [int(get_number_id(each)) for each in test_names]

categories_train = df.iloc[category_list_train,9].tolist()
titles_train = df.iloc[category_list_train,3].tolist()

categories_test = df.iloc[category_list_test,9].tolist()
title_test = df.iloc[category_list_test,3].tolist()


doc_vec_train={}
genres_train={}
for i in range(len(titles_train)):
    doc_vec_train[titles_train[i]]= train_vectors[i]
    genres_train[titles_train[i]] = categories_train[i]


tsne.plot_vectors(model_google,doc_vec_train,genres_train)