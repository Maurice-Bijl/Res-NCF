'''
Created on March 14, 2020
Keras/TensorFlow 2.0 Implementation of Neural Personalized Ranking (NeuPR) recommender model in:
"Going deeper with One-class Collaborative Filtering Systems"

@author: Maurice Bijl

** original code from Song, B. et al. 2018. "Neural collaborative ranking"
'''

from keras.regularizers import l2
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, Lambda, Multiply, Concatenate,BatchNormalization, Activation

def NeuPR(num_users, num_items, mf_dim=10, reg_mf=0, layers=[20,20,20], reg_layers=[0,0,0]):
    assert len (layers) == len (reg_layers)
    num_layer = len (layers)

    user_input = Input (shape = (1,), dtype = 'int32')
    item_input_pos = Input (shape = (1,), dtype = 'int32')
    item_input_neg = Input (shape = (1,), dtype = 'int32')

    MF_embedding_user = Embedding (input_dim = num_users, output_dim = mf_dim, embeddings_initializer = 'random_normal',
                                   name = 'mf_embedding_user', embeddings_regularizer = l2 (reg_mf), input_length = 1)
    MF_embedding_item = Embedding (input_dim = num_items, output_dim = mf_dim, embeddings_initializer = 'random_normal',
                                   name = 'mf_embedding_item', embeddings_regularizer = l2 (reg_mf), input_length = 1)
    MLP_embedding_user = Embedding (input_dim = num_users, output_dim =  int (layers [0] / 2),
                                    embeddings_initializer = 'random_normal',
                                    name = 'mlp_embedding_user', embeddings_regularizer = l2 (reg_layers[0]),
                                    input_length = 1)
    MLP_embedding_item = Embedding (input_dim = num_items, output_dim =  int (layers [0] / 2),
                                    embeddings_initializer = 'random_normal',
                                    name = 'mlp_embedding_item', embeddings_regularizer = l2 (reg_layers[0]),
                                    input_length = 1)

    mf_user_latent = Flatten() (MF_embedding_user (user_input))
    mf_item_latent_pos = Flatten() (MF_embedding_item (item_input_pos))
    mf_item_latent_neg = Flatten() (MF_embedding_item (item_input_neg))

    prefer_pos = Multiply()([mf_user_latent, mf_item_latent_pos])
    prefer_neg = Multiply()([mf_user_latent, mf_item_latent_neg])
    prefer_neg = Lambda (lambda x: -x) (prefer_neg)
    mf_vector = Concatenate()([prefer_pos, prefer_neg])

    mlp_user_latent = Flatten () (MLP_embedding_user (user_input))
    mlp_item_latent_pos = Flatten () (MLP_embedding_item (item_input_pos))
    mlp_item_latent_neg = Flatten () (MLP_embedding_item (item_input_neg))
    mlp_item_latent_neg = Lambda (lambda x: -x) (mlp_item_latent_neg)
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent_pos, mlp_item_latent_neg])

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

    predict_vector = Concatenate()([mf_vector, mlp_vector])

    prediction = Dense (1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'prediction') (
        predict_vector)
    model = Model (inputs = [user_input, item_input_pos, item_input_neg],
                   outputs = prediction)

    return model

def NBPR(num_users, num_items, mf_dim=10, reg_mf=0):

    user_input = Input (shape = (1,), dtype = 'int32')
    item_input_pos = Input (shape = (1,), dtype = 'int32')
    item_input_neg = Input (shape = (1,), dtype = 'int32')

    MF_embedding_user = Embedding (input_dim = num_users, output_dim = mf_dim, embeddings_initializer = 'random_normal',
                                   name = 'user_embedding', embeddings_regularizer = l2 (reg_mf), input_length = 1)
    MF_embedding_item = Embedding (input_dim = num_items, output_dim = mf_dim, embeddings_initializer = 'random_normal',
                                   name = 'item_embedding', embeddings_regularizer = l2 (reg_mf), input_length = 1)

    mf_user_latent = Flatten() (MF_embedding_user (user_input))
    mf_item_latent_pos = Flatten() (MF_embedding_item (item_input_pos))
    mf_item_latent_neg = Flatten() (MF_embedding_item (item_input_neg))

    prefer_pos = Multiply()([mf_user_latent, mf_item_latent_pos])
    prefer_neg = Multiply()([mf_user_latent, mf_item_latent_neg])
    prefer_neg = Lambda (lambda x: -x) (prefer_neg)
    mf_vector = Concatenate()([prefer_pos, prefer_neg])

    prediction = Dense (1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'prediction') (
        mf_vector)
    model = Model (inputs = [user_input, item_input_pos, item_input_neg],
                   outputs = prediction)

    return model

def MLP_pair(num_users, num_items, layers=[20,20,20], reg_layers=[0,0,0]):
    assert len (layers) == len (reg_layers)
    num_layer = len (layers)

    user_input = Input (shape = (1,), dtype = 'int32')
    item_input_pos = Input (shape = (1,), dtype = 'int32')
    item_input_neg = Input (shape = (1,), dtype = 'int32')

    MLP_embedding_user = Embedding (input_dim = num_users, output_dim =  int (layers [0] / 2),
                                    embeddings_initializer = 'random_normal',
                                    name = 'user_embedding', embeddings_regularizer = l2 (reg_layers [0]),
                                    input_length = 1)
    MLP_embedding_item = Embedding (input_dim = num_items, output_dim =  int (layers [0] / 2),
                                    embeddings_initializer = 'random_normal',
                                    name = 'item_embedding', embeddings_regularizer = l2 (reg_layers [0]),
                                    input_length = 1)

    mlp_user_latent = Flatten () (MLP_embedding_user (user_input))
    mlp_item_latent_pos = Flatten () (MLP_embedding_item (item_input_pos))
    mlp_item_latent_neg = Flatten () (MLP_embedding_item (item_input_neg))
    mlp_item_latent_neg = Lambda (lambda x: -x) (mlp_item_latent_neg)
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent_pos, mlp_item_latent_neg])

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

    prediction = Dense (1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'prediction') (
        mlp_vector)
    model = Model (inputs = [user_input, item_input_pos, item_input_neg],
                   outputs = prediction)
    return model