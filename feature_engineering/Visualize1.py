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
import re 

book_pickle_path="../data/preprocessed/books_1635_preprocessed.pickle"
name_pickle_path = "../data/preprocessed/names_1635.pickle"

class Visualize:
    def init_data(self,book_pickle_path,name_pickle_path):
        print("Init data Started")
        self.RAW_BOOK_FOLDER = os.path.join("..","data_collection","raw_books")
        self.PREPROCESSED_STORE_LOC = os.path.join("..","data","preprocessed")
        self.google_corpus_path  = "../models/GoogleNews-vectors-negative300.bin"
        self.model_google = KeyedVectors.load_word2vec_format(self.google_corpus_path,binary=True)

        self.list_of_words = CommonHelpers.load_pickle(book_pickle_path)
        self.book_names =  CommonHelpers.load_pickle(name_pickle_path)
        print("Init data completed")

    def load_data(self):
        print("Loading data")
        self.book_vectors = FeatureEngineering.get_avg_feature_vectors(self.list_of_words,self.model_google,300)
        df = pd.read_csv("../data/pg_catalog.csv")
        df["category"] = df["LoCC"].str[:1]
        category_list_train = [int(self.get_number_id(each)) for each in self.book_names]
        selected_books = df[df["Text#"].isin(category_list_train)]
        self.categories = selected_books.iloc[:,9].tolist()
        self.titles = selected_books.iloc[:,3].tolist()
        print("Loading data complete")

    def save_pickles(self):
        CommonHelpers.dump_pickle("../data/feature/word2vec_google_1635_books.pickle",self.book_vectors)
        CommonHelpers.dump_pickle("../data/feature/categories.pickle",self.categories)
        CommonHelpers.dump_pickle("../data/feature/titles.pickle",self.titles)

    def load_pickles(self):
        self.book_vectors = CommonHelpers.load_pickle("../data/feature/word2vec_google_1635_books.pickle")
        self.categories = CommonHelpers.load_pickle("../data/feature/categories.pickle")
        self.titles = CommonHelpers.load_pickle("../data/feature/titles.pickle")

    def get_number_id(self,file_name):
        regex = r"([a-z_A-Z]+)(\d+)"
        match = re.search(regex, file_name)
        if match!=None:
            return match.group(2)

    def plot(self):
        doc_vec={}
        genres={}
        for i in range(len(self.titles)):
            doc_vec[self.titles[i]]= self.book_vectors[i]
            genres[self.titles[i]] = self.categories[i]
        print("Starting plot")
        tsne.plot_vectors(doc_vec,genres)

