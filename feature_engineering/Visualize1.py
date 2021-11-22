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

    def load_data(self):
        self.book_vectors = CommonHelpers.load_pickle("../data/feature/word2vec_google_books.pickle")
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

