import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot
import nltk
from gensim.models import Word2Vec, KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import spacy
import seaborn as sns
import pandas as pd
import plotly.express as px
import colorcet as cc
# nlp = spacy.load('en_core_web_sm')


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


def tsne_plot(labels,tokens,genre):
	"Creates and TSNE model and plots it"
	tsne_model = TSNE(perplexity=10, n_components=2,
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
	palette = sns.color_palette(cc.glasbey, n_colors=25)
	#plt.figure(figsize=(16, 16))
	sns.set_context('notebook', font_scale=1.1)
	sns.set_style("ticks")

	if genre is not None:
		sns.lmplot(data=df, x='X', y='Y', hue='Genre', fit_reg=False,
				legend=True,palette=palette)
	else:
		sns.lmplot(data=df, x='X', y='Y', fit_reg=False,
				legend=True,palette=palette)

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
	plt.legend(loc="lower left", ncol=len(df.columns))
	plt.show()
	print('hello')


def plot_words(model,words):
	'''
	words = {'adventure': ['adventure', 'voyage', 'combat', 'fight', 'battle', 'escapade', 'dangerous', 'undertake', 'gamble', 'chance', 'risk', 'hazard', 'venture', 'stake', 'jeopardy'],
				'romance': ['romance', 'love', 'affair', 'woo', 'solicit', 'flirt', 'dally', 'butterfly', 'coquet', 'coquette', 'philander', 'comfort', 'console', 'solace', 'ease', 'soothe', 'friendship', 'passion', 'beloved', 'dear', 'honey', 'intimate', 'beautiful', 'marry', 'espouse', 'sweet', 'angelic', 'perfume', 'girl', 'miss', 'child', 'kiss', 'treasure'],
				'tragedy': ['tragedy', 'death', 'disaster', 'calamity', 'catastrophe', 'cataclysm', 'betrayal', 'treason', 'perfidy', 'grief', 'heartache', 'heartbreak', 'sorrow', 'humiliation', 'mortification', 'abasement', 'chagrin', 'misfortune', 'luck', 'omen', 'cry'],
				'mystery': ['suspense', 'frenzy', 'mystery', 'enigma', 'secret', 'whodunit', 'thriller', 'crime', 'detective', 'investigator', 'police', 'evidence', 'testify', 'witness', 'attest', 'clue', 'murder', 'execution', 'kill', 'weapon', 'fear', 'dread', 'terror', 'panic', 'pocket', 'solve'],
				'humor': ['comedy', 'drollery', 'clown', 'funny', 'humor', 'wit', 'mood', 'temper', 'satire', 'sarcasm', 'irony', 'remark', 'parody', 'lampoon', 'spoof', 'mockery', 'burlesque', 'travesty', 'pasquinade', 'laugh']}
	'''
	(labels,tokens,genre) = tsne_prep_from_words(model,words)
	tsne_plot(labels,tokens,genre)



def plot_vectors(doc_vecs,genre_dict):
	(labels,tokens,genre) = tsne_prep_from_vectors(doc_vecs,genre_dict)
	tsne_plot(labels,tokens,genre)

