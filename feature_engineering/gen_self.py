import Generate_word2vec_features

import numpy as np
import pandas as pd

gen = Generate_word2vec_features.Word2Vec_Features()

gen.load_data(model_type="self",cache=False)
train_vec =np.nan_to_num(gen.book_vectors)