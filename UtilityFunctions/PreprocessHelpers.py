import nltk
import re
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk import tokenize 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation
import pickle
import math
import string
import re
import xml.etree.ElementTree as ET
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

class Preprocessor:
    def __init__(self) -> None:
        self.stemmer=PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def set_text(self,raw_string):
        self.raw_string=raw_string
        self.clean_chars()

    def clean_chars(self):
        self.raw_string = s = re.sub('[^A-Za-z]', r' ', self.raw_string)

    def remove_proper_nouns(self,words):
        tagged_sentence = nltk.tag.pos_tag(words)
        edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
        return edited_sentence
    
    def tokenize_corpus(self): # tokenize on whitespace and remove punctuation
        words = []
        # todo check if all punctuations are removed
        sentences = nltk.tokenize.word_tokenize(self.raw_string)
        words =[each.lower() for each in sentences]
        return words

    def remove_punctuation(self,words):
        
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

    def remove_single_letters(self,words):
        cleaned = [each for each in words if each and len(each)>1]
        return cleaned

    
    def run_basic_pipeline(self,remove_punctuation=True,remove_proper_nouns=False):
        if self.raw_string is None:
            raise NotImplementedError
        tokenized_words = self.tokenize_corpus()
        cleaned_words = self.remove_stop_words(tokenized_words)
        if remove_punctuation is True:
            cleaned_words = self.remove_punctuation(cleaned_words)
        if remove_proper_nouns:
            cleaned_words = self.remove_proper_nouns(cleaned_words)
        stemmed = self.stem(cleaned_words)
        return stemmed

    def run_lemma_pipeline(self,remove_punctuation=True,remove_proper_nouns=False):
        if self.raw_string is None:
            raise NotImplementedError
        tokenized_words = self.tokenize_corpus()
        cleaned_words = self.remove_stop_words(tokenized_words)
        if remove_punctuation:
            cleaned_words = self.remove_punctuation(cleaned_words)
        if remove_proper_nouns:
            cleaned_words = self.remove_proper_nouns(cleaned_words)
        
        lemmed = self.lemma(cleaned_words)
        cleaned_lemmed = self.remove_single_letters( self.remove_stop_words(lemmed))
        return cleaned_lemmed