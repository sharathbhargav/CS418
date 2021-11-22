import sys,os
sys.path.append( os.path.join(".."))
from UtilityFunctions import CommonHelpers, PreprocessHelpers


RAW_BOOK_FOLDER = os.path.join("..","data_collection","raw_books")
PREPROCESSED_STORE_LOC = os.path.join("..","data","eda") # change output folder
aggragate_pickle_name = "books_english.pickle"
name_pickle_name = "names_english.pickle"


books=[]
book_names=[]
file_list = os.listdir(RAW_BOOK_FOLDER)
start=0
cnt=0
CommonHelpers.dump_pickle(os.path.join(PREPROCESSED_STORE_LOC,aggragate_pickle_name),books)
CommonHelpers.dump_pickle(os.path.join(PREPROCESSED_STORE_LOC,name_pickle_name),book_names)
for i in range(200,len(file_list),200):
    books = CommonHelpers.load_pickle(os.path.join(PREPROCESSED_STORE_LOC,aggragate_pickle_name))
    names = CommonHelpers.load_pickle(os.path.join(PREPROCESSED_STORE_LOC,name_pickle_name))
    for file_name in file_list[start:i]:
        cnt+=1
        full_text= CommonHelpers.load_pickle(os.path.abspath(os.path.join(RAW_BOOK_FOLDER, file_name)))
        preprocessor = PreprocessHelpers.Preprocessor()    
        preprocessor.set_text(full_text)
        words = preprocessor.run_eda_pipeline() # Change here to run_lemma_pipeline
        # print(file_name, len(words))
        books.append(words)
        book_names.append(file_name)
        if cnt%20==0:
            print(cnt)
    CommonHelpers.dump_pickle(os.path.join(PREPROCESSED_STORE_LOC,aggragate_pickle_name),books)
    CommonHelpers.dump_pickle(os.path.join(PREPROCESSED_STORE_LOC,name_pickle_name),book_names)
    print("Checkpoint ",i)
    start=i