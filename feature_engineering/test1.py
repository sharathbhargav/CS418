# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline


# %%
RAW_BOOK_FOLDER = os.path.join("..","data_collection","raw_books")
PREPROCESSED_STORE_LOC = os.path.join("..","data","preprocessed")


# %%
google_corpus_path  = "../models/GoogleNews-vectors-negative300.bin"
model_google = KeyedVectors.load_word2vec_format(google_corpus_path,binary=True)


# %%
list_of_words = CommonHelpers.load_pickle("../data/preprocessed/hundred_books_preprocessed.pickle")
book_names =  CommonHelpers.load_pickle("../data/preprocessed/hundred_names.pickle")

train_words = list_of_words[:90]
train_names = book_names[:90]
test_words  = list_of_words[90:]
test_names = book_names[90:]


# %%
train_vectors = FeatureEngineering.get_avg_feature_vectors(train_words,model_google,300)
test_vectors = FeatureEngineering.get_avg_feature_vectors(test_words,model_google,300)


# %%
"""
forest = RandomForestClassifier(n_estimators = 100)

print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_vectors,train_names)


# %%
result = forest.predict(test_vectors)

"""
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
"""

forest = RandomForestClassifier(n_estimators = 100)
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_vectors,categories_train)
result = forest.predict(test_vectors)
print(result)
"""