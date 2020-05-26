'''
Created on March 14, 2020
Keras/TensorFlow 2.0 Implementation of Residual Neural Matrix Factorization (ResMF) and
Residual Personalized Ranking (ResPR) recommender models in:
"Going deeper with One-class Collaborative Filtering Systems"

@author: Maurice Bijl
'''

#### import packages ---------------------------

from keras import backend as K
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Add, Flatten, Multiply, Concatenate, Lambda, \
    BatchNormalization, Activation

def init_normal(shape, dtype=None):
    return K.random_normal (shape, dtype = dtype)

# def ResMLP(res_layer, weights, reg_layers):
#     ''' Residual building blocks. When the final layer has a different shape than the identity block,
#     the shape of the idenity block is changed to the correct size with a dense layer'''
#     assert len (weights) == len (reg_layers), "weights and reg_layers do not have the same length"
#
#     for block in range (len (weights)):
#         shortcut = res_layer
#         layers = weights [block]
#         reg = reg_layers [block]
#         assert len (layers) == len (reg), "weights_%d and reg_layers_%d do not have the same length" % (block, block)
#         for index, weight in enumerate (layers):
#             res_layer = Dense (weight, kernel_regularizer = l2 (reg [index]),
#                                name = 'layer%d_%d' % (block, index)) (res_layer)
#             res_layer =  BatchNormalization(name = 'batch%d_%d' % (block, index)) (res_layer)
#             res_layer = Activation('relu')(res_layer)
#
#         ## identity block
#         if shortcut.shape [1] == res_layer.shape[1]: ## check if shapes are the same
#             res_layer = Add(name = "Add_%d" % block)([res_layer, shortcut])
#
#         ## dense block
#         else:
#             shortcut = Dense (res_layer.shape[1], kernel_regularizer = l2 (reg [len (layers) - 1]),
#                               name = 'dense_short%d' % block) (shortcut)
#             shortcut = BatchNormalization(name = 'batch_short%d' % block)(shortcut)
#             shortcut = Activation('relu')(shortcut)
#             res_layer = Add(name = "Add_%d" % block)([res_layer, shortcut])
#     return res_layer

def ResMLP(res_layer, weights, reg_layers):
    ''' Residual building blocks. When the final layer has a different shape than the identity block,
    the shape of the idenity block is changed to the correct size with a dense layer'''
    assert len (weights) == len (reg_layers), "weights and reg_layers do not have the same length"

    for block in range (len (weights)):
        shortcut = res_layer
        layers = weights [block]
        reg = reg_layers [block]
        assert len (layers) == len (reg), "weights_%d and reg_layers_%d do not have the same length" % (block, block)
        for index, weight in enumerate (layers):
            if index == (len(layers)-1):
                res_layer = Dense (weight, kernel_regularizer = l2 (reg [index]),
                                   name = 'layer%d_%d' % (block, index)) (res_layer)
                res_layer = BatchNormalization (name = 'batch%d_%d' % (block, index)) (res_layer)
                ## identity block
                if shortcut.shape [1] == res_layer.shape [1]:  ## check if shapes are the same
                    res_layer = Add (name = "Add_%d" % block) ([res_layer, shortcut])

                ## dense block
                else:
                    shortcut = Dense (res_layer.shape [1], kernel_regularizer = l2 (reg [len (layers) - 1]),
                                      name = 'dense_short%d' % block) (shortcut)
                    shortcut = BatchNormalization (name = 'batch_short%d' % block) (shortcut)
                    res_layer = Add (name = "Add_%d" % block) ([res_layer, shortcut])
                res_layer = Activation ('relu') (res_layer)

            else:
                res_layer = Dense (weight, kernel_regularizer = l2 (reg [index]),
                                   name = 'layer%d_%d' % (block, index)) (res_layer)
                res_layer =  BatchNormalization(name = 'batch%d_%d' % (block, index)) (res_layer)
                res_layer = Activation('relu')(res_layer)

    return res_layer

def ResMF(num_users, num_items, mf_dim=10, reg_mf=0, layers=[[20,20],[20, 10]], reg_layers=[[0,0],[0, 0]]):
    assert len(layers) == len(reg_layers)

    # Input variables
    user_input = Input (shape = (1,), dtype = 'int32', name = 'user_input')
    item_input = Input (shape = (1,), dtype = 'int32', name = 'item_input')

    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  embeddings_initializer = init_normal, embeddings_regularizer = l2(reg_mf),
                                  input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  embeddings_initializer = init_normal, embeddings_regularizer = l2(reg_mf),
                                  input_length=1)

    MLP_Embedding_User = Embedding (input_dim = num_users, output_dim = int (layers[0][0] / 2),
                                    name = 'mlp_embedding_user', embeddings_initializer = init_normal,
                                    embeddings_regularizer = l2 (reg_layers[0][0]), input_length = 1)
    MLP_Embedding_Item = Embedding (input_dim = num_items, output_dim = int (layers[0][0] / 2),
                                    name = 'mlp_embedding_item', embeddings_initializer = init_normal,
                                    embeddings_regularizer = l2 (reg_layers[0][0]), input_length = 1)

    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent]) # element-wise multiply

    # MLP part
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])

    ## create first layer of MLP
    first_layer = Dense (layers[0][1], kernel_regularizer = l2 (reg_layers[0][1]),
                        activation = 'relu', name = 'layer1') (mlp_vector)

    ## build model with shortcuts
    res_layers = layers[1:]
    res_reg = reg_layers[1:]
    mlp_vector = ResMLP (first_layer, res_layers, res_reg)

    # Concatenate MF and MLP parts
    predict_vector = Concatenate()([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense (1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = "prediction") (
        predict_vector)

    model = Model (inputs = [user_input, item_input],
                   outputs = prediction)
    return model


def ResPR(num_users, num_items, mf_dim=10, reg_mf=0, layers=[[20,20],[20, 10]], reg_layers=[[0,0],[0, 0]]):
    assert len (layers) == len (reg_layers)

    user_input = Input (shape = (1,), dtype = 'int32')
    item_input_pos = Input (shape = (1,), dtype = 'int32')
    item_input_neg = Input (shape = (1,), dtype = 'int32')

    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  embeddings_initializer = init_normal, embeddings_regularizer = l2(reg_mf),
                                  input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  embeddings_initializer = init_normal, embeddings_regularizer = l2(reg_mf),
                                  input_length=1)

    MLP_Embedding_User = Embedding (input_dim = num_users, output_dim = int (layers[0][0] / 2),
                                    name = 'mlp_embedding_user', embeddings_initializer = init_normal,
                                    embeddings_regularizer = l2 (reg_layers[0][0]), input_length = 1)
    MLP_Embedding_Item = Embedding (input_dim = num_items, output_dim = int (layers[0][0] / 2),
                                    name = 'mlp_embedding_item', embeddings_initializer = init_normal,
                                    embeddings_regularizer = l2 (reg_layers[0][0]), input_length = 1)

    mf_user_latent = Flatten() (MF_Embedding_User (user_input))
    mf_item_latent_pos = Flatten() (MF_Embedding_Item (item_input_pos))
    mf_item_latent_neg = Flatten() (MF_Embedding_Item (item_input_neg))

    prefer_pos = Multiply()([mf_user_latent, mf_item_latent_pos])
    prefer_neg = Multiply()([mf_user_latent, mf_item_latent_neg])
    prefer_neg = Lambda (lambda x: -x) (prefer_neg)
    mf_vector = Concatenate()([prefer_pos, prefer_neg])

    mlp_user_latent = Flatten () (MLP_Embedding_User (user_input))
    mlp_item_latent_pos = Flatten () (MLP_Embedding_Item (item_input_pos))
    mlp_item_latent_neg = Flatten () (MLP_Embedding_Item (item_input_neg))
    mlp_item_latent_neg = Lambda (lambda x: -x) (mlp_item_latent_neg)
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent_pos, mlp_item_latent_neg])

    ## create first layer of MLP
    first_layer = Dense (layers[0][1], kernel_regularizer = l2 (reg_layers[0][1]),
                        activation = 'relu', name = 'layer1') (mlp_vector)

    ## build model with shortcuts
    res_layers = layers[1:]
    res_reg = reg_layers[1:]
    mlp_vector = ResMLP (first_layer, res_layers, res_reg)

    predict_vector = Concatenate()([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense (1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'prediction') (
        predict_vector)
    model = Model (inputs = [user_input, item_input_pos, item_input_neg],
                   outputs = prediction)
    return model

def MLP_res_point(num_users, num_items, layers=[[20,20],[20, 10]], reg_layers=[[0,0],[0, 0]]):
    assert len(layers) == len(reg_layers)

    # Input variables
    user_input = Input (shape = (1,), dtype = 'int32', name = 'user_input')
    item_input = Input (shape = (1,), dtype = 'int32', name = 'item_input')

    MLP_Embedding_User = Embedding (input_dim = num_users, output_dim = int (layers[0][0] / 2), name = 'user_embedding',
                                    embeddings_initializer = init_normal,
                                    embeddings_regularizer = l2 (reg_layers[0][0]), input_length = 1)
    MLP_Embedding_Item = Embedding (input_dim = num_items, output_dim = int (layers[0][0] / 2), name = 'item_embedding',
                                    embeddings_initializer = init_normal,
                                    embeddings_regularizer = l2 (reg_layers[0][0]), input_length = 1)

    # MLP part
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])

    ## create first layer of MLP
    first_layer = Dense (layers[0][1], kernel_regularizer = l2 (reg_layers[0][1]),
                        activation = 'relu', name = 'layer1') (mlp_vector)

    ## build model with shortcuts
    res_layers = layers [1:]
    res_reg = reg_layers [1:]
    mlp_vector = ResMLP (first_layer, res_layers, res_reg)

    # Final prediction layer
    prediction = Dense (1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = "prediction") (
        mlp_vector)

    model = Model (inputs = [user_input, item_input],
                   outputs = prediction)
    return model

def MLP_res_pair(num_users, num_items, layers=[[20,20],[20, 10]], reg_layers=[[0,0],[0, 0]]):
    assert len (layers) == len (reg_layers)

    user_input = Input (shape = (1,), dtype = 'int32')
    item_input_pos = Input (shape = (1,), dtype = 'int32')
    item_input_neg = Input (shape = (1,), dtype = 'int32')

    MLP_Embedding_User = Embedding (input_dim = num_users, output_dim = int (layers[0][0] / 2), name = 'user_embedding',
                                    embeddings_initializer = init_normal,
                                    embeddings_regularizer = l2 (reg_layers[0][0]), input_length = 1)
    MLP_Embedding_Item = Embedding (input_dim = num_items, output_dim = int (layers[0][0] / 2), name = 'item_embedding',
                                    embeddings_initializer = init_normal,
                                    embeddings_regularizer = l2 (reg_layers[0][0]), input_length = 1)

    mlp_user_latent = Flatten () (MLP_Embedding_User (user_input))
    mlp_item_latent_pos = Flatten () (MLP_Embedding_Item (item_input_pos))
    mlp_item_latent_neg = Flatten () (MLP_Embedding_Item (item_input_neg))
    mlp_item_latent_neg = Lambda (lambda x: -x) (mlp_item_latent_neg)
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent_pos, mlp_item_latent_neg])

    ## create first layer of MLP
    first_layer = Dense (layers[0][1], kernel_regularizer = l2 (reg_layers[0][1]),
                        activation = 'relu', name = 'layer1') (mlp_vector)

    ## build model with shortcuts
    res_layers = layers [1:]
    res_reg = reg_layers [1:]
    mlp_vector = ResMLP (first_layer, res_layers, res_reg)

    prediction = Dense (1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'prediction') (
        mlp_vector)
    model = Model(inputs = [user_input, item_input_pos, item_input_neg],
                   outputs = prediction)
    return model