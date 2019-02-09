# -*- coding: utf-8 -*-

"""
Paraphrase detection using neural network
"""

import numpy as np

from keras import regularizers
from keras import backend as K
from keras import optimizers
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Lambda, Input, Embedding, Conv1D
from keras.layers import Concatenate, Reshape, Dense, Flatten
from keras.callbacks import Callback, EarlyStopping, TerminateOnNaN

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

import argparse

import sys

sys.path.append('..')

import utils

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(task_path, reverse_train, autogen):
    w2v_model, index_to_word, word_to_index, train_X_A, train_X_B, train_Y, \
            test_X_A, test_X_B, test_Y = utils.generate_dataset(task_path,
                                                                reverse_train,
                                                                autogen)
    max_seq_length = train_X_A.shape[1]
    logger.info("Max seq length:", max_seq_length)
    logger.info("X_train:", train_X_A.shape)
    logger.info("Y_train:", train_Y.shape)
    logger.info("X_test:", test_X_A.shape)
    logger.info("Y_test:", test_Y.shape)
    return (word_to_index, max_seq_length,
            train_X_A, train_X_B, train_Y,
            test_X_A, test_X_B, test_Y,
            w2v_model)


def build_model(wv_matrix, vec_dim, word_to_index, max_seq_length, filters,
                poolings, conv_act, dense_act, reg_value, hidden_units,
                trainable_embeddings):
    input_shape = (max_seq_length, )
    # Defining regularizer
    if reg_value is not None:
        regularizer = regularizers.l2(reg_value)
    else:
        regularizer = None

    # Generating encoder
    base_model = generate_encoder(wv_matrix, vec_dim, word_to_index,
                                  max_seq_length, filters, poolings, conv_act,
                                  regularizer)

    # Defining inputs
    input_X_A = Input(input_shape, name="input_X_A")
    input_X_B = Input(input_shape, name="input_X_B")

    # Encoding inputs
    s1_ga_pools = base_model(input_X_A)
    s2_ga_pools = base_model(input_X_B)

    use_euc = True
    use_cos = True
    use_abs = True
    feats = algo1(s1_ga_pools, s2_ga_pools, use_cos, use_euc, use_abs)

    X = Dense(hidden_units, name="fully_connected",
              kernel_regularizer=regularizer, activation=dense_act)(feats)
    X = Dense(2, name="output", kernel_regularizer=regularizer,
              activation="softmax")(X)

    siamese_net = Model(inputs=[input_X_A, input_X_B], outputs=X,
                        name="he_model_siamese")
    return siamese_net, base_model


def generate_encoder(wv_matrix, vec_dim, word_to_index, max_seq_length,
                     filters, poolings, conv_act, regularizer):
    """input shape: (None, max_seq_length)
       embedding_layer shape: (None, max_seq_length, embedding_dim)
       None refers to the minibatch size."""
    # Input layer
    input_X = Input((max_seq_length, ))

    # Embedding layer which transforms sequence of indices to embeddings.
    # It can be set to update the embeddings (trainable parameter).
    wv_layer = Embedding(len(word_to_index)+1,
                         vec_dim,
                         weights=[wv_matrix],
                         mask_zero=True,
                         input_length=max_seq_length)(input_X)
    block_a_output = block_a(wv_layer, filters, poolings, conv_act,
                             regularizer)
    model = Model(inputs=input_X,
                  outputs=block_a_output,
                  name='he_model_block_a')
    return model


def block_a(embedding_layer, filters, poolings, conv_act, regularizer):
    """ Returns a list with a tensor for each pooling function,
    where each tensor is of the shape:
    [?, len(filters), fnum]"""
    ###########
    # Block A #
    ###########
    outputs = []

    for i, Pool in enumerate(poolings):
        conv_out = []
        for fsize, fnum in filters:
            # if fsize == None do not do convolution
            if fsize:
                X = Conv1D(filters=fnum,
                           kernel_size=fsize,
                           strides=1,
                           padding="valid",
                           kernel_regularizer=regularizer,
                           activation=conv_act,
                           name="ga_conv_pool{}_ws{}".format(i, fsize)
                           )(embedding_layer)
                # X = BatchNormalization()(X)
                X = Pool(X)
            else:
                # NOTE: This will generate issues in the comparison
                # algorithms if fnum != embedding_dim
                X = Pool(embedding_layer)
            conv_out.append(Reshape((1, fnum))(X))

        if len(conv_out) == 1:
            outputs.append(conv_out[0])
        else:
            outputs.append(Concatenate(axis=1)(conv_out))

    return outputs


def algo1(s_A_pools, s_B_pools, use_cos, use_euc, use_abs):
    """:param s_A_pools: List of 'block_A' outputs of sentence A for different
                     pooling types [max, min, avg] where each entry has shape
                     (?, len(filters_ga), fnum_ga)
       :param s_B_pools: List of 'block_A' outputs of sentence B for different
                     pooling types [max, min, avg] where each entry has shape
                     (?, len(filters_ga), fnum_ga)
    """
    res = []
    i = 0
    for s_A, s_B in zip(s_A_pools, s_B_pools):
        # Vector norms of len(filters_ga)-dimensional vectors
        # s1_norm.shape = s2_norm.shape = (?, len(filters_ga), fnum_ga)
        s_A_norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(s_A)
        s_B_norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(s_B)

        sims = []

        if use_cos:
            # Cosine Similarity between vectors of shape (len(filters_ga),)
            # cos_sim.shape = (?, fnum_ga)
            cos_sim = Lambda(lambda x: K.sum(x[0]*x[1], axis=1),
                             name="a1_{}pool_cos".format(i)
                             )([s_A_norm, s_B_norm])
            sims.append(cos_sim)

        if use_euc:
            # Euclidean Distance between vectors of shape (len(filters_ga),)
            # euc_dis.shape = (?, fnum_ga)
            euc_dis = Lambda(lambda x: K.sqrt(K.clip(K.sum(K.square(x[0]
                                                           - x[1]), axis=1),
                                                     K.epsilon(), 1e+10)),
                             name="a1_{}pool_euc".format(i))([s_A, s_B])
            sims.append(euc_dis)

        if use_abs:
            # Absolute Distance between vectors of shape (len(filters_ga),)
            # abs_dis.shape = (?, len(filters_ga), fnum_ga)
            abs_dis = Lambda(lambda x: K.abs(x[0] - x[1]),
                             name="a1_ga_{}pool_abs".format(i)
                             )([s_A, s_B])
            sims.append(Flatten()(abs_dis))

        if len(sims) == 1:
            res.append(sims[0])
        else:
            res.append(Concatenate()([cos_sim, euc_dis]))
        i += 1

    # feah = (3 * 2 * fnum_ga)
    if len(res) == 1:
        feah = res[0]
    else:
        feah = Concatenate(name="feah")(res)
    return feah


def argument_parser():
    parser = argparse.ArgumentParser(description="""Multi-Perspective Sentence
            Similarity applied to Paraphrase Identification.
            The model is described in:
            He, H., Gimpel, K., Lin, J.J.:
            Multi-perspective sentence similarity modeling with convolutional
            neural networks.""")
    parser.add_argument("--rev-train",
                        dest="reversed_train",
                        action="store_true",
                        help="""Duplicate the training data size by adding
                             the reverse version of the pair of sentences.
                             Helpful when not using a Siamese model.""")
    parser.add_argument("--no-rev-train",
                        dest="reversed_train",
                        action="store_false")
    parser.add_argument("--autogen",
                        type=int,
                        dest="autogen",
                        help="Number of automatically generated negative \
                             samples.")

    parser.add_argument("--ngrams",
                        type=int,
                        dest="ngrams",
                        help="""Defines the filters sizes from 1 to the
                             assigned value. Eg. 1,2,3 in the case of
                             ngrams=3.""")
    parser.add_argument("--epochs",
                        type=int,
                        dest="epochs",
                        help="Number of training epochs.")
    parser.add_argument("--bsize",
                        type=int,
                        dest="batch_size",
                        help="Batch size used in gradient descent.")
    parser.add_argument("--train-emb",
                        dest="trainable_embeddings",
                        action="store_true",
                        help="Training the word embeddings as part of the \
                             model.")
    parser.add_argument("--no-train-emb",
                        dest="trainable_embeddings",
                        action="store_false")
    parser.add_argument("--hid-units",
                        type=int,
                        dest="hidden_units",
                        help="Hidden units used in the similarity layer.")
    parser.add_argument("--reg",
                        type=float,
                        dest="reg_value",
                        help="L2 regulatization value.")
    parser.add_argument("--optimizer",
                        type=str,
                        choices=("adam", "sgd"),
                        dest="optimizer",
                        help="Optimizer used during back propagation.")
    parser.add_argument("--learning-rate",
                        type=float,
                        dest="lr",
                        help="Learning rate used by the optimizer.")
    parser.add_argument("--loss",
                        type=str,
                        choices=("categorical_crossentropy",
                                 "categorical_hinge",
                                 "kullback_leibler_divergence"),
                        dest="loss",
                        help="Loss used during backpropagation.")
    parser.add_argument("--conv-act",
                        type=str,
                        dest="conv_act",
                        choices=("tanh", "relu", "sigmoid"),
                        help="Defines the activation functions of the \
                             convolutional layers")
    parser.add_argument("--dense-act",
                        type=str,
                        dest="dense_act",
                        choices=("tanh", "relu", "sigmoid"),
                        help="Defines the activation functions of the \
                             convolutional layers")

    parser.add_argument("--fnum-ba",
                        type=int,
                        dest="fnum_ba",
                        help="Number of holistic filters (Block A).")
    parser.add_argument("--inf-ba",
                        dest="use_inf_ba",
                        action="store_true",
                        help="""Defines if using the inf filter size in \
                             the holistic filters (Block A). When used, the \
                             number of filters must be the same as the \
                             embeddings size because computing similarity \
                             requires vectors of the same dimensions.""")
    parser.add_argument("--no-inf-ba",
                        dest="use_inf_ba",
                        action="store_false")
    parser.add_argument("--pool-ba",
                        nargs="+",
                        choices=("max", "min", "mean"),
                        dest="poolings",
                        help="Pooling operators used in group A.")

    parser.add_argument("--cos",
                        dest="use_cos",
                        action="store_true",
                        help="Defines if using cosine similarity in algorithm \
                             1.")
    parser.add_argument("--no-cos-a1",
                        dest="use_cos",
                        action="store_false")
    parser.add_argument("--euc",
                        dest="use_euc",
                        action="store_true",
                        help="Defines if using Euclidean distance in \
                             algorithm 1.")
    parser.add_argument("--no-euc",
                        dest="use_euc",
                        action="store_false")
    parser.add_argument("--abs",
                        dest="use_abs",
                        action="store_true",
                        help="Defines if using absolute difference in \
                             algorithm 1.")
    parser.add_argument("--no-abs",
                        dest="use_abs",
                        action="store_false")

    # Default parameters
    parser.set_defaults(
                        # Related to dataset generation
                        autogen=0,
                        reversed_train=False,

                        # Related to the model
                        ngrams=3,
                        epochs=10,
                        batch_size=32,
                        trainable_embeddings=True,
                        hidden_units=250,
                        reg_value=1e-4,
                        optimizer="adam",
                        lr=0.001,
                        loss="categorical_crossentropy",
                        conv_act="tanh",
                        dense_act="tanh",

                        # Group A
                        fnum_ba=525,
                        use_inf_ba=False,
                        poolings=["max"],
                        # Algorithm 1
                        use_cos=True,
                        use_euc=True,
                        use_abs=True,

                        # Others
                        print_encoder=False,
                        print_model=False,
                        verbosity=1)

    return parser.parse_args()


if __name__ == "__main__":
    print("==============\nPROGRAM STARTS\n==============")
    # Parsing command line arguments
    args = argument_parser()

    # Loading data and embeddings
    (word_to_index, max_seq_length,
     train_X_A, train_X_B, train_Y,
     test_X_A, test_X_B, test_Y,
     wv_model) = load_data(args)

    wv_matrix = np.zeros((len(word_to_index), wv_model.size), dtype=np.int32)
    for word, i in word_to_index.items():
        wv_matrix[i] = wv_model[word]

    # (None, args.holistic_fnum) represents the inf filter size.
    # There will be an error
    # if holistic_fnum != embeddings_size due to dimensions missmatching
    # during concatenation
    filters = [(i+1, args.fnum_ba) for i in range(args.ngrams)]
    if args.use_inf:
        assert wv_model.size == args.fnum_ba, \
               "When using --inf-ga, fnum_ga = embeddings_dim is needed."

        filters.append((None, args.fnum_ba))

    # Generating the pooling layers
    poolings = []
    if "max" in args.poolings:
        poolings.append(Lambda(lambda x: K.max(x, axis=1),
                        name="maxpool"))
    if "min" in args.poolings:
        poolings.append(Lambda(lambda x: K.min(x, axis=1),
                        name="minpool"))
    if "mean" in args.poolings:
        poolings.append(Lambda(lambda x: K.mean(x, axis=1),
                        name="meanpool"))

    # Generating the model
    print("================\nGenerating model\n================")
    print("Convolution activation: {}".format(args.conv_act))
    print("Dense activation: {}".format(args.dense_act))

    print("Block A config\n--------------")
    print("Filters:", filters)
    print("Pool ops:", args.poolings)
    print()
    print("Algorithm 1\n-----------")
    print("Cosine similarity:", args.use_cos)
    print("Euclidean distance:", args.use_euc)
    print("Absolute difference:", args.use_abs)
    print()

    siamese_model, encoder = build_model(wv_matrix, wv_model.size,
                                         word_to_index, max_seq_length,
                                         filters, poolings,
                                         args.conv_act, args.dense_act,
                                         args.reg_value, args.hidden_units,
                                         args.trainable_embeddings)

    trainable = int(np.sum([K.count_params(p)
                    for p in set(siamese_model.trainable_weights)]))
    non_trainable = int(np.sum([K.count_params(p)
                        for p in set(siamese_model.non_trainable_weights)]))
    print("Trainable parameters: {:,}".format(trainable))
    print("Non-trainable parameters: {:,}".format(non_trainable))
    print("Total parameters: {:,}".format(trainable + non_trainable))
    print("Done!")
    print("\n")

    # Printing summaries
    if args.print_encoder:
        encoder.summary()
    if args.print_model:
        siamese_model.summary()

    # Compiling model
    print("===============\nCompiling model\n===============")
    print("Optimizer:", args.optimizer)
    print("Learning rate:", args.lr)
    print("Loss:", args.loss)
    if args.optimizer == "adam":
        optimizer = optimizers.Adam(lr=args.lr)  # , clipnorm=1.)
    if args.optimizer == "sgd":
        optimizer = optimizers.SGD(lr=args.lr)  # , clipnorm=1.)
    siamese_model.compile(optimizer=optimizer,
                          loss=args.loss,
                          metrics=["accuracy"])
    print("Done!")
    print("\n")

    # Training model
    print("==================\nTraining model ...\n==================")
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    # Early stopping
    early_stopping_patience = 5
    early_stop = EarlyStopping(monitor='loss',
                               min_delta=0,
                               patience=early_stopping_patience,
                               verbose=0,
                               mode='auto')

    # Terminate training when NaN loss is encountered
    stop_nan = TerminateOnNaN()

    # Callbacks
    early_stop = False
    my_calls = [stop_nan]
    if early_stop:
        my_calls.append(early_stop)

    use_class_weight = False if args.autoneg == 0 else True
    if use_class_weight:
        if train_Y[train_Y == 1].size > train_Y[train_Y == 0].size:
            class_weight = {1: 1.0, 0: train_Y[train_Y == 1].size /
                            train_Y[train_Y == 0].size}
        else:
            class_weight = {1: train_Y[train_Y == 0].size /
                            train_Y[train_Y == 1].size, 0: 1.0}
        print("class_weight", class_weight)
    else:
        class_weight = None

    # Custom class to keep track of prediction history
    class PredHistory(Callback):
        def __init__(self):
            self.preds = []
            self.acchis = []
            self.f1his = []
            self.cmhis = []

        def on_epoch_end(self, epoch, logs=None):
            pred = self.model.predict([self.validation_data[0],
                                       self.validation_data[1]])
            self.preds.append(pred)
            predclass = np.argmax(pred, axis=1)
            goldstd = np.argmax(self.validation_data[2], axis=1)
            acc = accuracy_score(goldstd, predclass)
            print("val_acc: {:.4f}".format(acc))
            self.acchis.append(acc)
            f1 = f1_score(goldstd, predclass)
            print("val_f1: {:.4f}".format(f1))
            self.f1his.append(f1)
            cm = confusion_matrix(goldstd, predclass)
            print("val_conf_mat:\n{}".format(cm))
            self.cmhis.append(cm)

    perform_eval = PredHistory()
    my_calls.append(perform_eval)

    # Initial prediction with random initialization
    print("Forward propagation with random values")
    pred = siamese_model.predict([test_X_A, test_X_B])
    predclass = np.argmax(pred, axis=1)
    acc = accuracy_score(test_Y, predclass)
    print(acc)
    f1 = f1_score(test_Y, predclass)
    print(f1)
    cm = confusion_matrix(test_Y, predclass)
    print(cm)

    history = siamese_model.fit(x=[train_X_A, train_X_B],
                                y=to_categorical(train_Y),
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                validation_split=0.2,
                                class_weight=class_weight,
                                callbacks=my_calls,
                                verbose=args.verbosity)
    print("Done!")
    print("\n")
