import sys,os
sys.path.append( os.path.join(".."))
from UtilityFunctions import CommonHelpers, PreprocessHelpers


RAW_BOOK_FOLDER = os.path.join("..","data_collection","raw_books")
PREPROCESSED_STORE_LOC = os.path.join("..","data","preprocessed")

ten_books=[]
for file_name in os.listdir(RAW_BOOK_FOLDER):
    full_text= CommonHelpers.load_pickle(os.path.abspath(os.path.join(RAW_BOOK_FOLDER, file_name)))
    preprocessor = PreprocessHelpers.Preprocessor()    
    preprocessor.set_text(full_text)
    words = preprocessor.run_basic_pipeline(True)
    ten_books.append(words)
CommonHelpers.dump_pickle(os.path.join(PREPROCESSED_STORE_LOC,"ten_books_preprocessed.pickle"),ten_books)