import pickle

def load_resnet_embeddings(path="data/resnet18_features_train.pkl"):

    with open(path, mode="rb") as f:
        data = pickle.load(f)

    return data