from semantic_embedding_model import SemanticEmbeddingModel
from load_resnet import load_resnet_embeddings
from mygrad.nnet.losses import margin_ranking_loss
from mynn.optimizers.sgd import SGD
import numpy as np
import mygrad as mg

def train(training_triples, margin=0.5):
    caption_ids, good_img_ids, bad_img_ids = zip(*training_triples)

    caption_ids = np.array(caption_ids)
    good_img_ids = np.array(good_img_ids)
    bad_img_ids = np.array(bad_img_ids)
    print(good_img_ids.shape)

    model = SemanticEmbeddingModel()
    optim = SGD(model.parameters, learning_rate=0.1)

    resnet_embeddings = load_resnet_embeddings()

    for epoch_cnt in range(5):
        idxs = np.arange(len(caption_ids))
        np.random.shuffle(idxs)  

        caption_ids = caption_ids[idxs]
        good_img_ids = good_img_ids[idxs]
        bad_img_ids = bad_img_ids[idxs]

        for caption_id, good_id, bad_id in zip(caption_ids, good_img_ids, bad_img_ids):

            batch_caption = np.zeros(50)
            batch_good_img = resnet_embeddings[good_id]
            batch_bad_img = resnet_embeddings[bad_id]

            y_pred_good = mg.matmul(batch_caption, model(batch_good_img))
            y_pred_bad =  mg.matmul(batch_caption, model(batch_bad_img))

            loss = margin_ranking_loss(y_pred_good, y_pred_bad, 1, margin=margin)
            # back-propagate through your computational graph through your loss
            loss.backward()
            # execute gradient descent by calling step() of `optim`
            optim.step()
            # null your gradients (please!)
            loss.null_gradients()

train([(0,0,0),(0,0,0)])