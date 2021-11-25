from gensim.models import Word2Vec, KeyedVectors
import sys,os
sys.path.append( os.path.join(".."))
from UtilityFunctions import CommonHelpers,PreprocessHelpers,FeatureEngineering
import re
import pandas as pd
class Word2Vec_Features:
    def __init__(self):
        self.book_pickle_path="../data/preprocessed/books_english.pickle"
        self.name_pickle_path = "../data/preprocessed/names_english.pickle"

    def init_data(self):
        print("Init data Started")
        self.RAW_BOOK_FOLDER = os.path.join("..","data_collection","raw_books")
        self.PREPROCESSED_STORE_LOC = os.path.join("..","data","preprocessed")
        self.google_corpus_path  = "../models/GoogleNews-vectors-negative300.bin"
        self.list_of_words = CommonHelpers.load_pickle(self.book_pickle_path)
        self.book_names =  CommonHelpers.load_pickle(self.name_pickle_path)
        print("Init data completed")

    def load_data(self,cache=True):
        if cache == False:
            print("Loading data")
            self.model_google = KeyedVectors.load_word2vec_format(self.google_corpus_path,binary=True)
            self.book_vectors = FeatureEngineering.get_avg_feature_vectors(self.list_of_words,self.model_google,300)
            df = pd.read_csv("../data/pg_catalog.csv")
            df["category"] = df["LoCC"].str[:1]
            category_list_train = [int(self.get_number_id(each)) for each in self.book_names]
            selected_books = df[df["Text#"].isin(category_list_train)]
            self.categories=[]
            self.titles=[]
            for each in category_list_train:
                self.categories.append(selected_books[selected_books["Text#"]==each]["category"].values[0])
                self.titles.append(selected_books[selected_books["Text#"]==each]["Title"].values[0])
            print("Loading data complete")
            CommonHelpers.dump_pickle("../data/feature/word2vec_google_books.pickle",self.book_vectors)
            CommonHelpers.dump_pickle("../data/feature/categories.pickle",self.categories)
            CommonHelpers.dump_pickle("../data/feature/titles.pickle",self.titles)
            print("Saving to pickles done")
        else:
            self.book_vectors = CommonHelpers.load_pickle("../data/feature/word2vec_google_books.pickle")
            self.categories = CommonHelpers.load_pickle("../data/feature/categories.pickle")
            self.titles = CommonHelpers.load_pickle("../data/feature/titles.pickle")

    def get_number_id(self,file_name):
        regex = r"([a-z_A-Z]+)(\d+)"
        match = re.search(regex, file_name)
        if match!=None:
            return match.group(2)


