import nltk
import re
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk import tokenize 
from nltk.stem import PorterStemmer
from string import punctuation
import pickle
import math
nltk.download('punkt')
import xml.etree.ElementTree as ET
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
import string


class Preprocessor:
    def __init__(self) -> None:
        self.stemmer=PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def set_text(self,raw_string):
        self.raw_string=raw_string

    def tokenize_corpus(self): # tokenize on whitespace and remove punctuation
        words = []
        # todo check if all punctuations are removed
        sentences = nltk.tokenize.word_tokenize(self.raw_string)
        words =[each.lower() for each in sentences]
        return words

    def remove_punctuation(self,words):
        table = str.maketrans('', '', string.punctuation)
        stripped = [w for w in words if w not in string.punctuation]
        return stripped

    def remove_stop_words(self,words):
        cleaned = [each  for each in words if (each not in self.stop_words)]
        return cleaned

    def stem(self,words):
        stemmed=[ self.stemmer.stem(each) for each in words]
        return stemmed

    def lemma(self,words):
        lem = [self.lemmatizer.lemmatize(each) for each in words]
        return lem

    def run_basic_pipeline(self,remove_punctuation=False):
        if self.raw_string is None:
            raise NotImplementedError
        tokenized_words = self.tokenize_corpus()
        cleaned_words = self.remove_stop_words(tokenized_words)
        if remove_punctuation is True:
            cleaned_words = self.remove_punctuation(cleaned_words)
        stemmed = self.stem(cleaned_words)
        return stemmed

    def run_lemma_pipeline(self,remove_punctuation=False):
        if self.raw_string is None:
            raise NotImplementedError
        tokenized_words = self.tokenize_corpus()
        cleaned_words = self.remove_stop_words(tokenized_words)
        print("Aaa")
        if remove_punctuation:
            cleaned_words = self.remove_punctuation(cleaned_words)
        lemmed = self.lemma(cleaned_words)
        return lemmed