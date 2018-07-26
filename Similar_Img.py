import numpy as np

def similarity_finder(cap, k, database):
    '''
    Finds the top k similar images given a caption
    
    ####Parameters####
    cap: Caption that user wants to find an image to
    k: number of k images the function should find close to cap
    database: dictionary of image ids and descriptors

    ####Return####
    Returns k images in order of greatest to least similarity to cap
    '''
    descriptors = np.array(list(database.values()))
    img_cap_relation = np.dot(cap,descriptors)
    new_dict = dict(zip(database.keys(),img_cap_relation))
    sorted_correlation = sorted(new_dict.items(), key=lambda kv: kv[1])
    return list(sorted_correlation.keys()[0:k])
