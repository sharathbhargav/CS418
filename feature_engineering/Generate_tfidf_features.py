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
        self.unigram_tdifd_vectors = self.vectorizer.fit_transform(self.combined_books)
        print('create_unigram_vectors done')


    def create_unigram_vectors(self):
        self.vectorizer = TfidfVectorizer(analyzer = 'word', token_pattern = r'\w{1,}', ngram_range=(2, 2), stop_words='english', max_features = 200000)
        self.bigram_tdifd_vectors = self.vectorizer.fit_transform(self.combined_books)
        print('create_bigram_vectors done')


    def dim_reduce_svd_unigram(self):
        self.svd = TruncatedSVD(n_components=300, n_iter=7, random_state=42)
        self.svd_unigram_tfidf_vectors = self.svd.fit_transform(self.unigram_tdifd_vectors)
        print('dim_reduce_svd done')

    def dim_reduce_svd_bigram(self):
        self.svd = TruncatedSVD(n_components=300, n_iter=7, random_state=42)
        self.svd_bigram_tfidf_vectors = self.svd.fit_transform(self.bigram_tdifd_vectors)
        print('dim_reduce_svd done')
        
    def save_unigram_pickles(self, type = 'full'):
        if type == 'SVD':
            CommonHelpers.dump_pickle("../data/feature/svd_unigram_tfidf_vectors.pickle",self.svd_unigram_tfidf_vectors)
        else:
            CommonHelpers.dump_pickle("../data/feature/unigram_tfidf_vectors.pickle",self.unigram_tfidf_vectors)

        print('unigram save_pickles done')


    def save_bigram_pickles(self, type = 'full'):
        if type == 'SVD':
            CommonHelpers.dump_pickle("../data/feature/svd_bigram_tfidf_vectors.pickle",self.svd_bigram_tfidf_vectors)
        else:
            CommonHelpers.dump_pickle("../data/feature/bigram_tfidf_vectors.pickle",self.bigram_tfidf_vectors)

        print('unigram save_pickles done')



    def load_unigram_pickles(self, type = 'full'):
        if type == 'SVD':
            self.svd_unigram_tfidf_vectors = CommonHelpers.load_pickle("../data/feature/svd_unigram_tfidf_vectors.pickle")
        else:
            self.unigram_tfidf_vectors = CommonHelpers.load_pickle("../data/feature/unigram_tfidf_vectors.pickle")
        print('unigrams vector loaded')


    def save_bigram_pickles(self, type = 'full'):
        if type == 'SVD':
            self.svd_bigram_tfidf_vectors = CommonHelpers.load_pickle("../data/feature/svd_bigram_tfidf_vectors.pickle")
        else:
            self.bigram_tfidf_vectors = CommonHelpers.load_pickle("../data/feature/bigram_tfidf_vectors.pickle")
        print('bigrams vector loaded')



 