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
