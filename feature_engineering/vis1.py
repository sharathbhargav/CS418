import sys,os
sys.path.append( os.path.join(".."))
from UtilityFunctions import CommonHelpers,PreprocessHelpers
import numpy as np
book_pickle_path="../data/preprocessed/books_1635_preprocessed.pickle"
name_pickle_path = "../data/preprocessed/names_1635.pickle"

from Visualize1 import Visualize
vis = Visualize()
vis.load_pickles()
vis.book_vectors = np.nan_to_num(vis.book_vectors)
vis.plot()