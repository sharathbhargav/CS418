import numpy as np

def make_feature_vector(words, model, num_features):
    """
    Average the word vectors for a set of words
    """
    feature_vec = np.zeros((num_features,),dtype="float32")  # pre-initialize (for speed)
    n_words = 0
    index2word_set = set(model.index_to_key)  # words known to the model

    for word in words:
        if word in index2word_set: 
            n_words = n_words + 1
            feature_vec = np.add(feature_vec,model[word])
    
    feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


def get_avg_feature_vectors(books, model, num_features):
    """
    Calculate average feature vectors for all books
    """
    counter = 0
    feature_vectors = np.zeros((len(books),num_features), dtype='float32')  # pre-initialize (for speed)
    
    for review in books:
        feature_vectors[counter] = make_feature_vector(review, model, num_features)
        counter = counter + 1
    return feature_vectors