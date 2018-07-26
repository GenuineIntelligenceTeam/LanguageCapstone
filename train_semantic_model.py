from semantic_embedding_model import SemanticEmbeddingModel
from load_resnet import load_resnet_embeddings
from mygrad.nnet.losses import margin_ranking_loss
from mynn.optimizers.SGD import SGD
import numpy as np
import mygrad as mg

def train(training_triples, margin=0.5, batch_size=100):
    caption_ids, good_img_ids, bad_img_ids = zip(*training_triples)

    model = SemanticEmbeddingModel()
    optim = SGD(model.parameters, learning_rate=0.1)

    resnet_embeddings = load_resnet_embeddings()

    for epoch_cnt in range(5):
        idxs = np.arange(len(caption_ids))
        np.random.shuffle(idxs)  
        
        for batch_cnt in range(0, len(caption_ids)//batch_size):
            batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
            
            batch_captions = caption_embeddings[batch_indices]
            batch_good_imgs = resnet_embeddings[good_img_ids]
            batch_bad_imgs = resnet_embeddings[bad_img_ids]

            y_pred_good = mg.matmul(batch_captions, model(batch_good_imgs))
            y_pred_bad =  mg.matmul(batch_captions, model(batch_bad_imgs))

            loss = margin_ranking_loss(y_pred_good, y_pred_bad, 1, margin=margin)
            # back-propagate through your computational graph through your loss
            loss.backward()
            # execute gradient descent by calling step() of `optim`
            optim.step()
            # null your gradients (please!)
            loss.null_gradients()