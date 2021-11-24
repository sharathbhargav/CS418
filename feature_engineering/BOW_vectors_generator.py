import Generate_bow_features

gen = Generate_bow_features.BoW_Vectors()
gen.load_data()
gen.create_vectors()
gen.dim_reduce_svd()
#gen.dim_reduce_pca()
gen.save_pickles(type='SVD')
#gen.save_pickles(type='PCA')
gen.save_pickles()
