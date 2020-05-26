'''
Created on March 14, 2020
Keras/TensorFlow 2.0 Implementation of NeuMF*, NeuPR** and newly developed ResMF/ResPR recommender models in:
"Going deeper with One-class Collaborative Filtering Systems"

@author: Maurice Bijl

* original code (Theano implementation) from He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.
** original code from Song, B. et al. 2018. "Neural collaborative ranking"
'''
#### import packages ---------------------------
from time import time
import numpy as np
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
import datetime
import argparse
import os

## import custom modules
from DataLoader import DataSet
from TrainData import get_train_pair, get_train_point
from Models.NeuMF import MLP_point, NeuMF, GMF
from Models.NeuPR import MLP_pair, NeuPR, NBPR
from Models.ResidualModels import MLP_res_pair, MLP_res_point, ResMF, ResPR
from Evaluate.Pointwise import evaluate_pointwise
from Evaluate.Pairwise import evaluate_pairwise
from LoadPretrain import load_pretrain_model, load_pretrain_res

def parse_args():
    parser = argparse.ArgumentParser(description="Run different state-of-the-art recommender systems: "
                                                 "NeuMF, NeuPR, ResMF, ResPR")
    parser.add_argument('--dataset', default="NF_300K",
                        help="Select the name of the dataset")
    parser.add_argument('--path', default="data/",
                        help="Select the path of the processed data")
    parser.add_argument('--raw_path', default = "raw_data/",
                        help = "Select the path of the raw data")
    parser.add_argument('--filename', default = "combined_data_",
                        help = "Select the name of the raw data")
    parser.add_argument('--number_f', default = 4, type = int,
                        help = "Select the number of the raw data files")
    parser.add_argument ('--num_u', type = int, default = 5_000,
                         help = "number of users for subsample, 0 is all the users")
    parser.add_argument ('--num_i', type = int, default = 2_000,
                         help = "number of items for subsample, 0 is all the items")
    parser.add_argument('--model', default = "NeuPR",
                        help = '''select one of the following models: NeuMF, NeuPR, ResMF, ResPR, GMF, NBPR, 
                        MLP_point, MLP_pair, MLP_res_pair or MLP_res_point''')
    parser.add_argument('--mf_dim', default = 16, type = int,
                        help = "Select the size of the embedding layers of the linear part of the model")
    parser.add_argument('--layers', default = "[32,128,64,16]", type = str,
                        help = """MLP layers. Note that in case of non residual models the first layer is the 
                               concatenation of user and item embeddings. So layers[0]/2 is the embedding size. 
                               In case of the res. learning models the first first layer of the first list is the 
                               size of concatenation of user and item embeddings. layers[0][0]/2 is embedding size. 
                               layers[0][1] is the size of the first hidden layer.""")
    parser.add_argument('--reg_layers', default = "[0,0,0,0]", type = str,
                        help = "Regularization for each MLP layer. reg_layers[0] or reg_layers[0][0] (res. models)"
                               " is the regularization for embeddings.")
    parser.add_argument('--reg_mf', type = float, default = 0,
                        help = 'Regularization for MF embeddings.')
    parser.add_argument('--topK', type = float, default = 10,
                        help = 'number of recommendations for hr and ndcgs')
    parser.add_argument('--l_rate', type = float, default = 0.0001,
                        help = 'Learning rate')
    parser.add_argument('--learner', default = "adam",
                        help = 'choose model learner')
    parser.add_argument('--out', type=int, default=1,
                        help="Whether to save the trained model and results. 0: don't save")
    parser.add_argument('--epochs', type=int, default=15,
                        help="Number of epochs")
    parser.add_argument('--neg', type=int, default=4,
                        help="Number of negative instances to pair with a positive instance")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size")
    parser.add_argument('--verbose', type=int, default=1,
                        help="Show performance per X iterations")
    parser.add_argument('--pretrain_lin', default="",
                        help="Specify the pretrain model file for linear part. If empty, no pretrain will be used")
    parser.add_argument('--pretrain_non', default="",
                        help="Specify the pretrain model file for nonlinear part. If empty, no pretrain will be used")
    return parser.parse_args ()

def createFolder(directory):
    try:
        if not os.path.exists (directory):
            os.makedirs (directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

if __name__ == '__main__':
    args = parse_args ()
    dataset_name = args.dataset
    path = args.path
    raw = args.raw_path
    filename = args.filename
    number_f = args.number_f
    mf_dim = args.mf_dim
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    reg_mf = args.reg_mf
    topK = args.topK
    users = args.num_u
    items = args.num_i
    learning_rate = args.l_rate
    learner = args.learner
    out = args.out
    num_epochs = args.epochs
    num_negatives = args.neg
    batch_size = args.batch_size
    verbose = args.verbose
    pretrain_linear = args.pretrain_lin
    pretrain_nonlinear = args.pretrain_non

    model_init = eval(args.model)
    date = str(datetime.datetime.now ().date ())
    pointwise_models = ["NeuMF", "GMF","ResMF","MLP_point", "MLP_res_point"]

    #model_out_file = 'Pretrain/%s_%s_%d_%s_%s.h5' % (dataset_name, args.model, mf_dim, layers, date)
    ## create folder for model and results
    folder_name = 'Results/%s_%s_%d_%s_%s/' % (dataset_name, args.model, mf_dim, layers, date)
    createFolder (folder_name)
    print ("NeuMF arguments: %s " % (args))

    t1 = time ()
    # Loading data
    dataset = DataSet ()
    dataset.set_paths(path, raw, filename , number_f)
    dataset.loadClicks (10, 10, userSample = users, itemSample = items)
    train, validRatings, testRatings, testNegatives = dataset.trainMatrix, dataset.validRatings, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print ("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
           % (time () - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    if args.model in ["NeuMF", "NeuPR", "ResPR", "ResMF"]:
        model = model_init(num_users, num_items, mf_dim,reg_mf, layers, reg_layers)
    elif args.model in ["MLP_point", "MLP_pair","MLP_res_point", "MLP_res_pair"]:
        model = model_init (num_users, num_items, layers, reg_layers)
    else:
        model = model_init (num_users, num_items, mf_dim, reg_mf)

    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

    print(model.summary())

    # Load pretrain model
    if args.model == "NeuMF" and pretrain_linear != '' and pretrain_nonlinear != '':
        gmf_model = GMF(num_users, num_items, mf_dim)
        gmf_model.load_weights (pretrain_linear)
        mlp_model = MLP_point(num_users, num_items, layers, reg_layers)
        mlp_model.load_weights (pretrain_nonlinear)
        model = load_pretrain_model (model, gmf_model, mlp_model, len (layers))
        print ("Load pretrained GMF (%s) and MLP (%s) models done. " % (pretrain_linear, pretrain_nonlinear))

    if args.model == "NeuPR" and pretrain_linear != '' and pretrain_nonlinear != '':
        nbpr_model = NBPR(num_users, num_items, mf_dim)
        nbpr_model.load_weights (pretrain_linear)
        mlp_model = MLP_pair(num_users, num_items, layers, reg_layers)
        mlp_model.load_weights (pretrain_nonlinear)
        model = load_pretrain_model (model, nbpr_model, mlp_model, len (layers))
        print ("Load pretrained NBPR (%s) and MLP (%s) models done. " % (pretrain_linear, pretrain_nonlinear))

    if args.model == "ResPR" and pretrain_linear != '' and pretrain_nonlinear != '':
        nbpr_model = NBPR(num_users, num_items, mf_dim)
        nbpr_model.load_weights (pretrain_linear)
        mlp_model = MLP_res_pair(num_users, num_items, layers, reg_layers)
        mlp_model.load_weights (pretrain_nonlinear)
        model = load_pretrain_res(model, nbpr_model, mlp_model, layers)
        print ("Load pretrained NBPR (%s) and MLP_res (%s) models done. " % (pretrain_linear, pretrain_nonlinear))

    if args.model == "ResMF" and pretrain_linear != '' and pretrain_nonlinear != '':
        gmf_model = GMF(num_users, num_items, mf_dim, reg_mf)
        gmf_model.load_weights (pretrain_linear)
        mlp_model = MLP_res_point(num_users, num_items, layers, reg_layers)
        mlp_model.load_weights (pretrain_nonlinear)
        model = load_pretrain_res(model, gmf_model, mlp_model, layers)
        print ("Load pretrained GMF (%s) and MLP_res (%s) models done. " % (pretrain_linear, pretrain_nonlinear))

    # Init performance
    if args.model in pointwise_models:
        (hits, ndcgs) =evaluate_pointwise(model, validRatings, testNegatives, topK)
        (hits_test, ndcgs_test) = evaluate_pointwise (model, testRatings, testNegatives, topK)
        hits_test, ndcgs_test = np.array (hits_test), np.array (ndcgs_test)
        best_test_hr, best_test_ndcg = hits_test [hits_test >= 0].mean (), ndcgs_test [ndcgs_test >= 0].mean ()
    else:
        (hits, ndcgs) = evaluate_pairwise(model, validRatings, testNegatives, topK)
        (hits_test, ndcgs_test) = evaluate_pairwise (model, testRatings, testNegatives, topK)
        hits_test, ndcgs_test = np.array (hits_test), np.array (ndcgs_test)
        best_test_hr, best_test_ndcg = hits_test [hits_test >= 0].mean (), ndcgs_test [ndcgs_test >= 0].mean ()

    hr, ndcg = np.array (hits).mean (), np.array (ndcgs).mean ()

    print ('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if out > 0:
        model.save_weights (folder_name + 'model.h5', overwrite = True)

    all_loss = []
    val_hr = []
    val_ndcgs = []
    test_hr = []
    test_ndcgs = []
    if args.model in pointwise_models:
        for epoch in range (num_epochs):
            t1 = time ()
            # Generate training instances
            user_input, item_input, labels = get_train_point (train, num_negatives)

            # Training
            hist = model.fit ([np.array (user_input), np.array (item_input)],  # input
                              np.array (labels),  # labels
                              batch_size = batch_size, epochs = 1, verbose = 1, shuffle = True, )
            t2 = time ()
            # Evaluation
            if epoch % verbose == 0:

                # validate model
                (hits, ndcgs) = evaluate_pointwise (model, validRatings, testNegatives, topK)
                hits, ndcgs = np.array (hits), np.array (ndcgs)
                vhr, vndcg = hits [hits >= 0].mean (), ndcgs [ndcgs >= 0].mean ()
                val_hr.append(vhr)
                val_ndcgs.append(vndcg)

                # test model
                (hits, ndcgs) = evaluate_pointwise (model, testRatings, testNegatives, topK)
                hits, ndcgs = np.array (hits), np.array (ndcgs)
                hr, ndcg, loss = hits [hits >= 0].mean (), ndcgs [ndcgs >= 0].mean (), hist.history ['loss'] [0]
                all_loss.append (loss)
                test_hr.append (hr)
                test_ndcgs.append (ndcg)
                print ('Iteration %d/%d [%.1f s]: [Valid HR = %.4f, NDCG = %.4f, loss=%.6f]\t[Test HR = %.4f, '
                       'NDCG = %.4f], [%.1f s]' % (epoch + 1, num_epochs, t2 - t1, vhr, vndcg, loss, hr, ndcg,
                                                   time () - t2))

                if vhr > best_hr:
                    best_hr, best_ndcg, best_iter = vhr, vndcg, epoch
                    best_test_hr, best_test_ndcg = hr, ndcg
                    if out > 0:
                        model.save_weights (folder_name + 'model.h5', overwrite = True)
    else:
        for epoch in range (num_epochs):
            t1 = time ()
            # Generate training instances
            user_input, item_input_pos, item_input_neg, labels = get_train_pair(train, num_negatives)

            # Training
            # Training
            hist = model.fit ([np.array (user_input), np.array (item_input_pos), np.array (item_input_neg)],  # input
                              np.array (labels),  # labels
                              batch_size = batch_size, epochs = 1, verbose = verbose, shuffle = True)
            t2 = time ()
            # Evaluation
            if epoch % verbose == 0:

                # validate model
                (hits, ndcgs) = evaluate_pairwise (model, validRatings, testNegatives, topK)
                hits, ndcgs = np.array (hits), np.array (ndcgs)
                vhr, vndcg = hits [hits >= 0].mean (), ndcgs [ndcgs >= 0].mean ()
                val_hr.append(vhr)
                val_ndcgs.append(vndcg)

                # test model
                (hits, ndcgs) = evaluate_pairwise (model, testRatings, testNegatives, topK)
                hits, ndcgs = np.array (hits), np.array (ndcgs)
                hr, ndcg, loss = hits [hits >= 0].mean (), ndcgs [ndcgs >= 0].mean (), hist.history ['loss'] [0]
                all_loss.append (loss)
                test_hr.append (hr)
                test_ndcgs.append (ndcg)
                print ('Iteration %d/%d [%.1f s]: [Valid HR = %.4f, NDCG = %.4f, loss=%.6f]\t[Test HR = %.4f, '
                       'NDCG = %.4f], [%.1f s]' % (epoch +1, num_epochs, t2 - t1, vhr, vndcg, loss, hr, ndcg,
                                                   time () - t2))
                if vhr > best_hr:
                    best_hr, best_ndcg, best_iter = vhr, vndcg, epoch
                    best_test_hr, best_test_ndcg = hr, ndcg
                    if out > 0:
                        model.save_weights (folder_name + 'model.h5', overwrite = True)

    print ("End. Best Iteration %d: Test HR = %.4f, NDCG = %.4f. " % (best_iter, best_test_hr, best_test_ndcg))
    print ('learning_rate: %.5f , num_factor: %d' % (learning_rate, mf_dim))

    if out > 0:
        np.save (folder_name + 'train_loss.npy', np.array (all_loss))
        np.save (folder_name + 'val_hr.npy', np.array (val_hr))
        np.save (folder_name + 'val_ndcgs.npy', np.array (val_ndcgs))
        np.save (folder_name + 'test_hr.npy', np.array (test_hr))
        np.save (folder_name + 'test_ndcgs.npy', np.array (test_ndcgs))

        conclusion = "Best Iteration %d: \n" \
                     "Validation HR = %.4f, validation NDCG = %.4f.  \n" \
                     "Test HR = %.4f, NDCG = %.4f. " % (best_iter, val_hr [best_iter], val_ndcgs [best_iter],
                                                        best_test_hr, best_test_ndcg)

        text_file = open (folder_name + "conclusion.txt", "wt")
        n = text_file.write (conclusion)
        text_file.close ()