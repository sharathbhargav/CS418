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


    def load_data(self,model_type="google",cache=True):
        if cache == False:
            print("Loading data")
            print("Init data Started")
            self.RAW_BOOK_FOLDER = os.path.join("..","data_collection","raw_books")
            self.PREPROCESSED_STORE_LOC = os.path.join("..","data","preprocessed")
            self.google_corpus_path  = "../models/GoogleNews-vectors-negative300.bin"
            self.self_model_path = "../models/word2vec_std_file"
            self.list_of_words = CommonHelpers.load_pickle(self.book_pickle_path)
            self.book_names =  CommonHelpers.load_pickle(self.name_pickle_path)
            print("Init data completed")
            if model_type =="google":
                self.model_path = self.google_corpus_path
                self.model_google = KeyedVectors.load_word2vec_format(self.model_path,binary=True)
                self.num_of_features=300
            else:
                self.model_path=self.self_model_path
                self.model_google = KeyedVectors.load(self.model_path,mmap='r')
                self.num_of_features=100
            self.book_vectors = FeatureEngineering.get_avg_feature_vectors(self.list_of_words,self.model_google,self.num_of_features)
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
            if model_type =="google":
                CommonHelpers.dump_pickle("../data/feature/word2vec_google_books.pickle",self.book_vectors)
            else:
                CommonHelpers.dump_pickle("../data/feature/word2vec_w2v_gen_books.pickle",self.book_vectors)
            CommonHelpers.dump_pickle("../data/feature/categories.pickle",self.categories)
            CommonHelpers.dump_pickle("../data/feature/titles.pickle",self.titles)
            print("Saving to pickles done")
        else:
            if model_type =="google":
                self.book_vectors = CommonHelpers.load_pickle("../data/feature/word2vec_google_books.pickle")
            else:
                self.book_vectors = CommonHelpers.load_pickle("../data/feature/word2vec_w2v_gen_books.pickle")
            self.categories = CommonHelpers.load_pickle("../data/feature/categories.pickle")
            self.titles = CommonHelpers.load_pickle("../data/feature/titles.pickle")

    def get_number_id(self,file_name):
        regex = r"([a-z_A-Z]+)(\d+)"
        match = re.search(regex, file_name)
        if match!=None:
            return match.group(2)


