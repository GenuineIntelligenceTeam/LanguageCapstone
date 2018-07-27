import json
from PIL import Image
import urllib.request
import numpy as np

with open('captions_train2014.json') as json_file:
    data = json.load(json_file)


def displayImages(image_id):
    """ displays the image associated with the given ID


    Parameters
    ----------
    image_id : int
        the ID associated with the image that will be displayed
    """

    ids, jpgs = [], []
    for i in range(len(data['images'])):
        ids.append((data['images'][i])['id'])
        jpgs.append((data['images'][i])['flickr_url'])
    display_image_dict = dict(zip(ids, jpgs))
    image = display_image_dict[image_id]
    with urllib.request.urlopen(image) as url:
        with open('display.jpg', 'wb') as f:
            f.write(url.read())
    img = Image.open('display.jpg')
    img.show()


#create separate lists for image IDs and caption IDs
image_ids, ids = [], []
for i in range(len(data['annotations'])):
    image_ids.append((data['annotations'][i])['image_id'])
    ids.append((data['annotations'][i])['id'])

def getCaptionIDs(image_id):
    """ provides all 5 captions for image associated with the given ID


    Parameters
    ----------
    image_id : int
        the ID associated with the image that will be displayed

    Returns
    -------
    captions_IDs : list
        list of all 5 caption IDs of image
    """

    img_indices = [i for i, x in enumerate(image_ids) if x == image_id]
    captions_IDs = [ids[j] for j in img_indices]
    return captions_IDs

def generateData(n, k):
    """ generates dataset for training


    Parameters
    ----------
    n : int
        the number of good image IDs used for generating triples
    k : int
        the number of bad image IDs generated for every good image ID

    Returns
    -------
    triples : list of tuples
        list of tuples (caption_id, good_img_id, bad_img_id) for training the neural network
    """
    triples, caption_id, bad_img_id = [], [], []
    good_img_id = np.random.choice(image_ids,n)
    for id in good_img_id:
        bad_img_id += list(np.random.choice(list(set(image_ids)-set(good_img_id)), k))
        for i in range(k):
            caption_id.append(np.random.choice(getCaptionIDs(id)))
    triples = list(zip(caption_id, good_img_id, bad_img_id))

    return triples
