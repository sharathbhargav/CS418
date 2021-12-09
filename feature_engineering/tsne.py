import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot
import nltk
from gensim.models import Word2Vec, KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# import spacy
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import cdist


def tsne_prep_from_words(model,text_dict):
	labels = []
	tokens = []
	genre = []
	for key, values in text_dict.items():
		for word in values:
			if word in model:
				tokens.append(model[word])
				# tokens.append(nlp.vocab[word].vector) #spacy  using fasttext
				labels.append(word)
				genre.append(key)
	return (labels,tokens,genre)

def tsne_prep_from_vectors(vector_dict,genre_dict):
	labels=[]
	tokens=[]
	genre=[]
	for key, values in vector_dict.items():
		labels.append(key)
		tokens.append(values)
		if genre_dict is not None:
			genre.append(genre_dict[key])
	if genre_dict is not None:
		return (labels,tokens,genre)
	else:
		return (labels,tokens,None)
def cosine(a1,a2):
	val_out = 1 - cdist(a1, a2, 'cosine')
	return val_out

def tsne_plot(labels,tokens,genre):
	"Creates and TSNE model and plots it"
	tsne_model = TSNE(perplexity=10, n_components=2,metric=cosine,
						init='pca', n_iter=8500, random_state=1507)
	new_values = tsne_model.fit_transform(tokens)

	x = []
	y = []
	for value in new_values:
		x.append(value[0])
		y.append(value[1])

	df = pd.DataFrame()
	df['X'] = x
	df['Y'] = y
	if genre is not None:
		df['Genre'] = genre
	df["label"] = labels
	colors = ["#000000","#78909C","#3E2723","#DD2C00","#AEEA00",
				"#33691E","#64FFDA","#01579B","#311B92","#E040FB",
				"#C51162","#D50000","ffbb33","#00c851","#37b5e5",
				"#9933cc","#4A148C","148c4a","4a148c","8c4a14",
				"8c174a"
]

	sns.set_context('notebook', font_scale=1.1)
	sns.set_style("ticks")

	if genre is not None:
		sns.lmplot(data=df, x='X', y='Y', hue='Genre', fit_reg=False,
				legend=True)
	else:
		sns.lmplot(data=df, x='X', y='Y', fit_reg=False,
				legend=True)

	for i in range(len(x)):
		plt.scatter(x[i], y[i])
		'''
		plt.annotate(genre[i],
						xy=(x[i], y[i]),
						xytext=(5, 2),
						textcoords='offset points',
						ha='right',
						va='bottom',
						fontsize=8)
						'''
	plt.legend(loc="upper left", ncol=len(df.columns))
	plt.show()


def plot_words(model,words):
	
	(labels,tokens,genre) = tsne_prep_from_words(model,words)
	tsne_plot(labels,tokens,genre)



def plot_vectors(doc_vecs,genre_dict):
	(labels,tokens,genre) = tsne_prep_from_vectors(doc_vecs,genre_dict)
	tsne_plot(labels,tokens,genre)

