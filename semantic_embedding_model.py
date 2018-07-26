from mynn.layers.dense import dense
from mynn.initializers.glorot_normal import glorot_normal

class SemanticEmbeddingModel(object):

    INPUT_SIZE = 512
    OUTPUT_SIZE = 50

    def __init__(self):
        self.dense_layer = dense(INPUT_SIZE, OUTPUT_SIZE, weight_initializer=glorot_normal)

    def __call__(self, x):
        return self.dense_layer(x)

    @property
    def parameters(self):
        return self.dense_layer.parameters

