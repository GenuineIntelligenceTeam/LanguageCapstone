import json
from PIL import Image
import urllib.request

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

def getCaptions(image_id):
    """ provides all 5 captions for image associated with the given ID


    Parameters
    ----------
    image_id : int
        the ID associated with the image that will be displayed

    Returns
    -------
    final_captions : list
        list of all 5 captions of image
    """
    image_ids, ids, captions = [], [], []
    for i in range(len(data['annotations'])):
        image_ids.append((data['annotations'][i])['image_id'])
        ids.append((data['annotations'][i])['id'])
        captions.append((data['annotations'][i])['caption'])
    caption_dict = dict(zip(ids, captions))
    img_indices = [i for i, x in enumerate(image_ids) if x == image_id]
    final_ids = [ids[j] for j in img_indices]
    final_captions = [caption_dict[k] for k in final_ids]
    return final_captions
