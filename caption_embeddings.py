import numpy as np
import pickle
from collections import defaultdict
import numpy as np
import time
import gensim
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
path = "./data/glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)
import re, string
from collections import Counter

# this creates a regular expression that identifies all punctuation character
# don't include this in `strip_punc`, otherwise you will re-compile this expression
# every time you call the function


def to_counter(doc):
    """
    Produce word-count of document, removing all punctuation
    and removing all punctuation.

    Parameters
    ----------
    doc : str

    Returns
    -------
    collections.Counter
        lower-cased word -> count"""

    return Counter(strip_punc(doc).lower().split())


def to_bag(counters, k=None, stop_words=None):
    """
    [word, word, ...] -> sorted list of top-k unique words
    Excludes words included in `stop_words`

    Parameters
    ----------
    counters : Iterable[Iterable[str]]

    k : Optional[int]
        If specified, only the top-k words are returned

    stop_words : Optional[Collection[str]]
        A collection of words to be ignored when populating the bag
    """
    bag = Counter()
    for counter in counters:
        bag.update(counter)

    if stop_words is not None:
        for word in set(stop_words):
            bag.pop(word, None)  # if word not in bag, return None
    return sorted(i for i,j in bag.most_common(k))



punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))



with open("./data/stopwords.txt", 'r') as r:
    stops = []
    for line in r:
        stops += [i.strip() for i in line.split('\t')]


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


def to_idf(bag, counters):
    """
    Given the bag-of-words, and the word-counts for each document, computes
    the inverse document-frequency (IDF) for each term in the bag.

    Parameters
    ----------
    bag : Sequence[str]
        Ordered list of words that we care about

    counters : Iterable[collections.Counter]
        The word -> count mapping for each document.

    Returns
    -------
    numpy.ndarray
        An array whose entries correspond to those in `bag`, storing
        the IDF for each term `t`:
                           log10(N / nt)
        Where `N` is the number of documents, and `nt` is the number of
        documents in which the term `t` occurs.
    """
    N = len(counters)
    nt = [sum(1 if t in counter else 0 for counter in counters) for t in bag]
    nt = np.array(nt, dtype=float)
    return np.log10(N / nt)


def caption_to_word_embedding (captions, use_stop_words = False):
    word_index_dict = {}
    word_counts = [to_counter(doc) for doc in captions]
    bag = to_bag(word_counts, stop_words=None)
    for i in range(len(bag)):
        word_index_dict[bag[i]] = i
    idf = to_idf(bag,word_counts)
    wordemb = np.zeros((len(captions),50))
    for i in range(len(captions)):
        captions[i] = strip_punc(captions[i])
        caption_as_list = list(captions[i].split())
        for word in caption_as_list:
            wordemb[i] += glove[word] * idf[word_index_dict[word]]
    return wordemb
