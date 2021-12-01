
import Generate_tfidf_features

gen = Generate_tfidf_features.TFIDF_Vectors()
gen.load_data()
gen.create_unigram_vectors()
gen.create_bigram_vectors()

gen.save_unigram_pickles()
gen.save_bigram_pickles()




