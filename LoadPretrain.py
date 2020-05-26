
#### import packages ---------------------------
import numpy as np

def load_pretrain_model(model, linear_model, nonlinear_model, num_layers):
    # linear embeddings
    linear_user_embeddings = linear_model.get_layer ('user_embedding').get_weights ()
    linear_item_embeddings = linear_model.get_layer ('item_embedding').get_weights ()
    model.get_layer ('mf_embedding_user').set_weights (linear_user_embeddings)
    model.get_layer ('mf_embedding_item').set_weights (linear_item_embeddings)

    # non-linear embeddings
    mlp_user_embeddings = nonlinear_model.get_layer ('user_embedding').get_weights ()
    mlp_item_embeddings = nonlinear_model.get_layer ('item_embedding').get_weights ()
    model.get_layer ('mlp_embedding_user').set_weights (mlp_user_embeddings)
    model.get_layer ('mlp_embedding_item').set_weights (mlp_item_embeddings)

    # MLP layers
    for i in range (1, num_layers):
        mlp_layer_weights = nonlinear_model.get_layer ('layer%d' % i).get_weights ()
        model.get_layer ('layer%d' % i).set_weights (mlp_layer_weights)

    # ## for models with batch normalization
        mlp_layer_batch = nonlinear_model.get_layer ('batch%d' % i).get_weights ()
        model.get_layer ('batch%d' % i).set_weights (mlp_layer_batch)

    # Prediction weights
    linear_prediction = linear_model.get_layer ('prediction').get_weights ()
    mlp_prediction = nonlinear_model.get_layer ('prediction').get_weights ()
    new_weights = np.concatenate ((linear_prediction [0], mlp_prediction [0]), axis = 0)
    new_b = linear_prediction [1] + mlp_prediction [1]
    model.get_layer ('prediction').set_weights ([0.5 * new_weights, 0.5 * new_b])
    return model

def load_pretrain_res(model, linear_model, nonlinear_model, weights):
    # linear embeddings
    linear_user_embeddings = linear_model.get_layer ('user_embedding').get_weights ()
    linear_item_embeddings = linear_model.get_layer ('item_embedding').get_weights ()
    model.get_layer ('mf_embedding_user').set_weights (linear_user_embeddings)
    model.get_layer ('mf_embedding_item').set_weights (linear_item_embeddings)

    # non-linear embeddings
    mlp_user_embeddings = nonlinear_model.get_layer ('user_embedding').get_weights ()
    mlp_item_embeddings = nonlinear_model.get_layer ('item_embedding').get_weights ()
    model.get_layer ('mlp_embedding_user').set_weights (mlp_user_embeddings)
    model.get_layer ('mlp_embedding_item').set_weights (mlp_item_embeddings)

    # set weights for first layer
    mlp_layer = nonlinear_model.get_layer ('layer1').get_weights ()
    model.get_layer ('layer1').set_weights (mlp_layer)

    weights = weights[1:]
    # set weights of residual blocks
    for block in range (len (weights)):
        shortcut = mlp_layer
        layers = weights [block]
        for index in range(len(layers)):
            mlp_layer = nonlinear_model.get_layer ('layer%d_%d' % (block, index)).get_weights ()
            model.get_layer ('layer%d_%d' % (block, index)).set_weights (mlp_layer)
            mlp_layer_batch = nonlinear_model.get_layer ('batch%d_%d' % (block, index)).get_weights ()
            model.get_layer ('batch%d_%d' % (block, index)).set_weights (mlp_layer_batch)

        if shortcut[1].shape[0] != mlp_layer[1].shape[0]:
            shortcut = nonlinear_model.get_layer ('dense_short%d' % block).get_weights ()
            model.get_layer ('dense_short%d' % block).set_weights (shortcut)
            mlp_layer_batch = nonlinear_model.get_layer ('batch_short%d' % block).get_weights ()
            model.get_layer ('batch_short%d' % block).set_weights (mlp_layer_batch)

    # Prediction weights
    linear_prediction = linear_model.get_layer ('prediction').get_weights ()
    mlp_prediction = nonlinear_model.get_layer ('prediction').get_weights ()
    new_weights = np.concatenate ((linear_prediction [0], mlp_prediction [0]), axis = 0)
    new_b = linear_prediction [1] + mlp_prediction [1]
    model.get_layer ('prediction').set_weights ([0.5 * new_weights, 0.5 * new_b])
    return model