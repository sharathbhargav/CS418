from nltk.cluster import KMeansClusterer
import nltk
import os
import sys
import RunModels

sys.path.append( os.path.join(".."))
from UtilityFunctions import CommonHelpers,PreprocessHelpers,FeatureEngineering
import Generate_word2vec_features
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN, MeanShift, OPTICS
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE

class Kmeans:
        def __init__(self,all_vectors,all_labels): 
                self.NUM_CLUSTERS = 21

                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(all_vectors, 
                all_labels, test_size=0.20, random_state=4)
                self.data =self.x_train

        def nltk_k(self):
                rn =np.random.random((21, 300))
                kclusterer = KMeansClusterer(21, distance=nltk.cluster.util.cosine_distance, repeats=25)
                self.assigned_clusters = kclusterer.cluster(self.x_train, assign_clusters=True)
                CommonHelpers.dump_pickle("../data/kmeans_clusters.pickle",assigned_clusters)
                print (self.assigned_clusters)


        def load_clusters(self):
                self.assigned_clusters=CommonHelpers.load_pickle("../data/kmeans_clusters.pickle")

                
        def plot(self):

                model = TSNE(n_components=2, random_state=0)
                np.set_printoptions(suppress=True)
                Y=model.fit_transform(self.x_train)
                colors = np.array(["r","g","b"])
                import itertools
                import seaborn as sns
                colors = itertools.cycle(["r", "b", "g"])
                sns.scatterplot(x=Y[:, 0],y=Y[:, 1],hue=self.assigned_clusters,palette=sns.color_palette("hls",21))
                # plt.scatter(Y[:, 0], Y[:, 1], c=self.assigned_clusters, colormap='jet',s=10,alpha=.5)

                for j in range(len(self.y_train)):    
                        plt.annotate(self.y_train[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
                
                plt.show()  
                print(set(self.assigned_clusters))    
gen = Generate_word2vec_features.Word2Vec_Features()
gen.init_data()
gen.load_data(cache=True)
train_vec =np.nan_to_num( gen.book_vectors)
titles= gen.titles
categories= gen.categories
kmeans = Kmeans(train_vec,categories)
# kmeans.nltk_k()
kmeans.load_clusters()
kmeans.plot()