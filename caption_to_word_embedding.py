import numpy as np
import pickle
from collections import defaultdict
import numpy as np
import time
import gensim
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
path = "/Users/caseygoldstein/Downloads/glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)
import re, string
from collections import Counter

with open("/Users/caseygoldstein/LanguageCapstone/idf.pkl", mode="rb") as opened_file:
    idf = pickle.load(opened_file)

with open("/Users/caseygoldstein/LanguageCapstone/word_index_dict.pkl",mode = 'rb') as opened_file:
    word_index_dict = pickle.load(opened_file)

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    """
     Removes all punctuation from a string.

        Parameters
        ----------
        corpus : str

        Returns
        -------
        str
            the corpus with all punctuation removed
    # substitute all punctuation marks with ""
    """
    return punc_regex.sub(' ', corpus)



def caption_to_word_embedding (captions, use_stop_words = False):
    wordemb = np.zeros((len(captions),50))


    for i in range(len(captions)):
        captions[i] = strip_punc(captions[i])
        caption_as_list = list(captions[i].split())
        for word in caption_as_list:
            wordemb[i] += glove[word] * idf[word_index_dict[word]]
    return wordemb





