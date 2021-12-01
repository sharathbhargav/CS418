import sys,os
sys.path.append( os.path.join(".."))
from UtilityFunctions import CommonHelpers, PreprocessHelpers, FeatureEngineering
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDF_Vectors:
    def __init__(self):
        self.PREPROCESSED_STORE_LOC = os.path.join("..","data","preprocessed")
        print(' __init__done')

    def load_data(self):
        self.all_books = CommonHelpers.load_pickle(os.path.join(self.PREPROCESSED_STORE_LOC,"books_english_preprocessed.pickle"))
        self.combined_books = [' '.join(book) for book in self.all_books]
        print('load_data done')

    def create_unigram_vectors(self):
        self.vectorizer = TfidfVectorizer(analyzer = 'word', token_pattern = r'\w{1,}', ngram_range=(1, 1), stop_words='english')
        self.unigram_vectors = self.vectorizer.fit_transform(self.combined_books)
        print('create_vectors done')

    def create_bigram_vectors(self):
        self.vectorizer = TfidfVectorizer(analyzer = 'word', token_pattern = r'\w{1,}', ngram_range=(2, 2), stop_words='english', max_features = 200000)
        self.bigram_vectors = self.vectorizer.fit_transform(self.combined_books)
        print('create_vectors done')
        
    def save_unigram_pickles(self):
        CommonHelpers.dump_pickle("../data/feature/tfidf_unigram_vectors.pickle",self.unigram_vectors)
        print('unigram save_pickles done')

    def save_bigram_pickles(self):
        CommonHelpers.dump_pickle("../data/feature/tfidf_bigram_vectors.pickle",self.bigram_vectors)
        print('bigram save_pickles done')


    def load_unigram_pickles(self):
        self.unigram_vectors = CommonHelpers.load_pickle("../data/feature/tfidf_unigram_vectors.pickle")
        print('unigrams vector loaded')


    def load_unigram_pickles(self):
        self.bigram_vectors = CommonHelpers.load_pickle("../data/feature/tfidf_bigram_vectors.pickle")
        print('bigrams vector loaded')



 