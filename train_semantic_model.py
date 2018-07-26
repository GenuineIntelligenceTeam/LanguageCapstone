from semantic_embedding_model import SemanticEmbeddingModel
from load_resnet import load_resnet_embeddings
from mygrad.nnet.losses import margin_ranking_loss
from mynn.optimizers.sgd import SGD
import numpy as np
import mygrad as mg

def process_triples(training_triples, caption_embeddings, resnet_embeddings):
    caption_ids, good_img_ids, bad_img_ids = zip(*training_triples)
    N = len(caption_ids)

    caption_desc = np.zeros((N, 50))
    for i, caption_id in enumerate(caption_ids):
        caption_desc[i,:] = caption_embeddings[caption_id]

    good_desc = np.zeros((N, 512))
    for i, img_id in enumerate(good_img_ids):
        good_desc[i,:] = resnet_embeddings[img_id]
    
    bad_desc = np.zeros((N, 512))
    for i, img_id in enumerate(bad_img_ids):
        bad_desc[i,:] = resnet_embeddings[img_id]

    return caption_desc, good_desc, bad_desc


def train(training_triples, margin=0.5):

    model = SemanticEmbeddingModel()
    optim = SGD(model.parameters, learning_rate=0.1)

    resnet_embeddings = load_resnet_embeddings()

    caption_desc, good_desc, bad_desc = process_triples(training_triples, caption_embeddings, resnet_embeddings)

    for epoch_cnt in range(5):
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

resnet_embeddings = load_resnet_embeddings()