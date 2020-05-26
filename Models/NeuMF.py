'''
Created on March 14, 2020
TensorFlow 2.0 Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
"Going deeper with One-class Collaborative Filtering Systems"

@author: Maurice Bijl

** original code (Theano implementation) from He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.
'''

## import keras packages
from keras import backend as K
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, Multiply, Concatenate, BatchNormalization, Activation

## functions
def init_normal(shape, dtype=None):
    return K.random_normal (shape, dtype = dtype)

def NeuMF(num_users, num_items, mf_dim=10, reg_mf=0, layers=[20,20,20], reg_layers=[0,0,0]):
    assert len (layers) == len (reg_layers)
    num_layer = len (layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input (shape = (1,), dtype = 'int32', name = 'user_input')
    item_input = Input (shape = (1,), dtype = 'int32', name = 'item_input')

    # Embedding layer
    MF_Embedding_User = Embedding (input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                   embeddings_initializer = init_normal,
                                   embeddings_regularizer = l2 (reg_mf), input_length = 1)
    MF_Embedding_Item = Embedding (input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                   embeddings_initializer = init_normal,
                                   embeddings_regularizer = l2 (reg_mf), input_length = 1)

    MLP_Embedding_User = Embedding (input_dim = num_users, output_dim = int (layers [0] / 2),
                                    name = "mlp_embedding_user", embeddings_initializer = init_normal,
                                    embeddings_regularizer = l2 (reg_layers [0]), input_length = 1)
    MLP_Embedding_Item = Embedding (input_dim = num_items, output_dim = int (layers [0] / 2),
                                    name = 'mlp_embedding_item', embeddings_initializer = init_normal,
                                    embeddings_regularizer = l2 (reg_layers [0]), input_length = 1)

    # MF part
    mf_user_latent = Flatten () (MF_Embedding_User (user_input))
    mf_item_latent = Flatten () (MF_Embedding_Item (item_input))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])  # element-wise multiply

    # MLP part
    mlp_user_latent = Flatten () (MLP_Embedding_User (user_input))
    mlp_item_latent = Flatten () (MLP_Embedding_Item (item_input))
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])

    #original model
    # for idx in range (1, num_layer):
    #     mlp_vector = Dense (layers [idx], kernel_regularizer = l2 (reg_layers [idx]),
    #                         activation = 'relu', name = 'layer%d' % idx) (mlp_vector)

    ## model with batch normalization before activation
    for idx in range (1, num_layer):
        mlp_vector = Dense (layers [idx], kernel_regularizer = l2 (reg_layers [idx]),
                            name = 'layer%d' % idx)(mlp_vector)
        mlp_vector = BatchNormalization(name = "batch%d" % idx)(mlp_vector)
        mlp_vector = Activation('relu')(mlp_vector)

    # Concatenate MF and MLP parts
    predict_vector =Concatenate()([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense (1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = "prediction") (
        predict_vector)

    model = Model (inputs = [user_input, item_input], outputs = prediction)

    return model

def GMF(num_users, num_items, mf_dim, reg_mf = 0 ):
    # Input variables
    user_input = Input (shape = (1,), dtype = 'int32', name = 'user_input')
    item_input = Input (shape = (1,), dtype = 'int32', name = 'item_input')

    MF_Embedding_User = Embedding (input_dim = num_users, output_dim = mf_dim, name = 'user_embedding',
                                   embeddings_initializer = init_normal, embeddings_regularizer = l2 (reg_mf),
                                   input_length = 1)
    MF_Embedding_Item = Embedding (input_dim = num_items, output_dim = mf_dim, name = 'item_embedding',
                                   embeddings_initializer = init_normal, embeddings_regularizer = l2 (reg_mf),
                                   input_length = 1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten () (MF_Embedding_User (user_input))
    item_latent = Flatten () (MF_Embedding_Item (item_input))

    # Element-wise product of user and item embeddings
    mf_vector = Multiply()([user_latent, item_latent])

    prediction = Dense (1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'prediction') (mf_vector)

    model = Model (inputs = [user_input, item_input], outputs = prediction)

    return model

def MLP_point(num_users, num_items, layers=[20,20,20], reg_layers=[0,0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len (layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input (shape = (1,), dtype = 'int32', name = 'user_input')
    item_input = Input (shape = (1,), dtype = 'int32', name = 'item_input')

    MLP_Embedding_User = Embedding (input_dim = num_users, output_dim = int (layers [0] / 2), name = 'user_embedding',
                                    embeddings_initializer = init_normal, embeddings_regularizer = l2 (reg_layers [0]),
                                    input_length = 1)
    MLP_Embedding_Item = Embedding (input_dim = num_items, output_dim = int (layers [0] / 2), name = 'item_embedding',
                                    embeddings_initializer = init_normal, embeddings_regularizer = l2 (reg_layers [0]),
                                    input_length = 1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten () (MLP_Embedding_User (user_input))
    item_latent = Flatten () (MLP_Embedding_Item (item_input))

    # The 0-th layer is the concatenation of embedding layers
    mlp_vector = Concatenate()([user_latent, item_latent])

    ##original model
    # for idx in range (1, num_layer):
    #     mlp_vector = Dense (layers [idx], kernel_regularizer = l2 (reg_layers [idx]),
    #                         activation = 'relu', name = 'layer%d' % idx) (mlp_vector)

    ## model with batch normalization before activation
    for idx in range (1, num_layer):
        mlp_vector = Dense (layers [idx], kernel_regularizer = l2 (reg_layers [idx]),
                            name = 'layer%d' % idx)(mlp_vector)
        mlp_vector = BatchNormalization(name = "batch%d" % idx)(mlp_vector)
        mlp_vector = Activation('relu')(mlp_vector)

    # Final prediction layer
    prediction = Dense (1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'prediction') (mlp_vector)

    model = Model(inputs = [user_input, item_input], outputs = prediction)

    return model