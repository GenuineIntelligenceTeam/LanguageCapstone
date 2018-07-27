from semantic_embedding_model import SemanticEmbeddingModel
from load_resnet import load_resnet_embeddings
from caption_embeddings import caption_to_word_embedding
from data import generate_data
from mygrad.nnet.losses import margin_ranking_loss
from mynn.optimizers.sgd import SGD
import numpy as np
import mygrad as mg

def process_triples(training_triples, caption_embeddings, resnet_embeddings):
    """ Performs preprocessing on input tuples to generate data compatible with a NN.

    Parameters
    ----------
    training_triples : List[Tuple(int, int, int)]
        List of all triples input to the neural network, in the form (caption_id, good_id, bad_id).

    caption_embeddings : dict{int -> List, shape=(50,)}
        Dictionary mapping caption ids to shape-(50,) embeddings.

    resnet_embeddings : dict{int -> mygrad.Tensor, shape=(512,)}
        Dictionary mapping image ids to shape-(512,) raw resnet embeddings.

    Returns
    -------
    caption_desc : numpy.ndarray, shape=(N, 50)
        2-dimensional array of correct caption embeddings

    good_desc : numpy.ndarray, shape=(N, 512)
        2-dimensional array of correct ("good") image descriptors

    bad_desc : numpy.ndarray, shape=(N, 512)
        2-dimensional array of incorrect ("bad") image descriptors
    """

    caption_ids, good_img_ids, bad_img_ids = zip(*training_triples)
    N = len(caption_ids)

    caption_desc = caption_embeddings(caption_ids)

    good_desc = np.zeros((N, 512))
    for i, img_id in enumerate(good_img_ids):
        good_desc[i,:] = resnet_embeddings[img_id]
    
    bad_desc = np.zeros((N, 512))
    for i, img_id in enumerate(bad_img_ids):
        bad_desc[i,:] = resnet_embeddings[img_id]

    return caption_desc, good_desc, bad_desc


def train(training_triples, margin=0.5, epochs=10):
    """ Trains a 512 by 50 linear classifier to fit given training data.
    
    Parameters
    ----------
    training_triples : List[Tuple(int, int, int)]
        List of all triples input to the neural network, in the form (caption_id, good_id, bad_id).

    margin : float, optional (default=0.5)
        Margin to be used in margin-ranking loss.

    epochs : int, optional (default=10)
        Number of epochs to train neural network for.
    """

    model = SemanticEmbeddingModel()
    optim = SGD(model.parameters, learning_rate=0.1)

    resnet_embeddings = load_resnet_embeddings()

    caption_desc, good_desc, bad_desc = process_triples(training_triples, caption_to_word_embedding, resnet_embeddings)

    print("processed triples")

    for epoch_cnt in range(epochs):
        idxs = np.arange(len(caption_desc))
        np.random.shuffle(idxs)  

        caption_desc = caption_desc[idxs]
        good_desc = good_desc[idxs]
        bad_desc = bad_desc[idxs]

        for i in range(len(caption_desc)):

            batch_caption = np.zeros(50)
            batch_good_img = good_desc[i,:]
            batch_bad_img = bad_desc[i,:]

            y_pred_good = mg.matmul(batch_caption, model(batch_good_img))
            y_pred_bad =  mg.matmul(batch_caption, model(batch_bad_img))

            loss = margin_ranking_loss(y_pred_good, y_pred_bad, 1, margin=margin)

            loss.backward()
            optim.step()
            loss.null_gradients()

train(generate_data(20, 5))