import numpy as np
import pandas as pd
import tensorflow as tf
import copy
import matplotlib
from matplotlib import pyplot as plt


class NN:
    def __init__(self, m, num_classes, h1, h2, lrn_rate, dropout, weight_decay,
                 x_train, y_train, x_test, y_test, blinded, blinded_path, id_to_labels_map, predict_blinded=False, seed=2017):
        # hyper-parameters
        self.m = m  # number of features
        self.class_count = num_classes
        self.h1 = h1
        self.h2 = h2
        self.lrn_rate = lrn_rate
        self.keep_prob = 1 - dropout
        self.weight_decay = weight_decay
        self.level = level
        self.seed = seed
        self.blinded = blinded
        self.blinded_path = blinded_path
        self.predict_blinded = predict_blinded
        self.id_to_label_map = id_to_labels_map
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # data
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # weights and biases
        self.weights = {}
        self.biases = {}
        self.output_before_act = None
        self.output = None

        # tf variables
        self.X = None
        self.Y = None
        self.kp_placeholder = None
        self.loss_optimizer = None
        self.trn_optimizer = None
        self.init_object = None
        self.create_model()
        self.train()

    def get_softmax_probs_only(self, sess, U):
        return sess.run(fetches=[self.output], feed_dict={self.X: self.x_train[U], self.Y: self.y_train[U]})[0]

    @staticmethod
    def select_bvsb_point(softmax_probs, U):
        max_prob_idx = np.argmax(softmax_probs, axis=1)
        max_probs = np.max(softmax_probs, axis=1)

        # finding the second highest prob
        sm_copy = np.array(copy.deepcopy(softmax_probs))
        sm_copy[np.arange(np.shape(sm_copy)[0]), max_prob_idx] = -np.inf
        second_highest_probs = np.max(sm_copy, axis=1)

        # get most uncertain point
        most_uncertain_idx = U[np.argmin(max_probs - second_highest_probs)]
        return most_uncertain_idx

    @staticmethod
    def select_entropy_point(softmax_probs, U):
        # Calculate entropies
        log_prob = np.log(softmax_probs)
        prod = softmax_probs * log_prob * -1
        entropies = np.sum(prod, axis=1)
        # Select point with highest entropy
        most_uncertain_idx = U[np.argmax(entropies)]
        return most_uncertain_idx

    @staticmethod
    def select_random_point(U):
        idx = np.random.randint(0, len(U))
        point = U[idx]
        return point

    def perform_single_trn_iteration(self, sess, L):
        sess.run(fetches=[self.trn_optimizer], 
                 feed_dict={self.X: self.x_train[L], self.Y: self.y_train[L], self.kp_placeholder: self.keep_prob})

    def get_data_metrics(self, sess, x_data, y_data):
        ce_loss, softmax_probs = sess.run(fetches=[self.loss_optimizer, self.output], feed_dict={self.X: x_data, self.Y: y_data})
        softmax_probs = np.array(softmax_probs)
        predicted_labels = np.argmax(softmax_probs, axis=1)
        true_labels = np.argmax(y_data, axis=1)
        acc = np.sum([x == y for x, y in zip(predicted_labels, true_labels)]) / len(predicted_labels)
        return round(ce_loss, 4), round(acc * 100, 4)

    def create_model(self):
        # standard tf computational graph creation
        # Creating tf placeholders
        self.X = tf.placeholder("float", [None, self.m])
        self.Y = tf.placeholder("int32", None)
        self.kp_placeholder = tf.placeholder("float")

        # layer 1
        self.weights['w1'] = tf.Variable(tf.random_normal(shape=[self.m, self.h1], seed=self.seed))
        self.biases['b1'] = tf.Variable(tf.random_normal(shape=[self.h1], seed=self.seed))
        before_act1 = tf.matmul(self.X, self.weights['w1']) + self.biases['b1']
        a1 = tf.nn.sigmoid(before_act1)
        tf.nn.dropout(a1, keep_prob=self.kp_placeholder)

        # layer 2
        self.weights['w2'] = tf.Variable(tf.random_normal(shape=[self.h1, self.h2], seed=self.seed))
        self.biases['b2'] = tf.Variable(tf.random_normal(shape=[self.h2], seed=self.seed))
        before_act2 = tf.matmul(a1, self.weights['w2']) + self.biases['b2']
        a2 = tf.nn.sigmoid(before_act2)
        tf.nn.dropout(a2, self.kp_placeholder)

        # layer 3 (output)
        self.weights['w3'] = tf.Variable(tf.random_normal(shape=[self.h2, self.class_count], seed=self.seed))
        self.biases['b3'] = tf.Variable(tf.random_normal(shape=[self.class_count], seed=self.seed))
        self.output_before_act = tf.matmul(a2, self.weights['w3']) + self.biases['b3']
        self.output = tf.nn.softmax(self.output_before_act)

        # metrics to be calculated
        regularizers = tf.nn.l2_loss(self.weights['w1']) + tf.nn.l2_loss(self.weights['w2']) + tf.nn.l2_loss(self.weights['w3'])
        ce_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.Y, logits=self.output_before_act))
        self.loss_optimizer = tf.reduce_mean(ce_loss + self.weight_decay * regularizers)
        optimizer = tf.train.AdamOptimizer(self.lrn_rate)
        self.trn_optimizer = optimizer.minimize(self.loss_optimizer)
        self.init_object = tf.global_variables_initializer()

    def single_tf_session(self, session_name):
        print('SESSION : ' + session_name)
        L = []
        U = list(np.arange(self.x_train.shape[0]))
        oracle_call = 0

        # result vectors
        trn_accs = []
        trn_losses = []
        test_accs = []
        test_losses = []

        with tf.Session() as sess:
            sess.run(self.init_object)
            while oracle_call < 2500:
                if session_name == 'ENTROPY' or session_name == 'BVSB':
                    # get softmax probabilites
                    softmax_probs = self.get_softmax_probs_only(sess, U)

                # selection of a point to be queried
                if session_name == 'BVSB':
                    most_uncertain_idx = self.select_bvsb_point(softmax_probs=softmax_probs, U=U)
                elif session_name == 'ENTROPY':
                    most_uncertain_idx = self.select_entropy_point(softmax_probs=softmax_probs, U=U)
                else:
                    # random learner
                    most_uncertain_idx = self.select_random_point(U=U)

                # remove this point from U and add it to L
                L.append(most_uncertain_idx)
                U.remove(most_uncertain_idx)
                oracle_call += 1

                # train the NN using only points in L
                self.perform_single_trn_iteration(sess=sess, L=L)

                # calculate training accuracy and cross entropy loss
                trn_loss, trn_acc = self.get_data_metrics(sess=sess, x_data=self.x_train[L], y_data=self.y_train[L])

                # calculate test accuracy and cross entropy loss
                test_loss, test_acc = self.get_data_metrics(sess=sess, x_data=self.x_test, y_data=self.y_test)

                trn_losses.append(trn_loss)
                trn_accs.append(trn_acc)
                test_losses.append(test_loss)
                test_accs.append(test_acc)

                if oracle_call % 100 == 0:
                    print("oracle call = %s, trn_loss = %s, trn_acc = %s, test_loss = %s, test_acc = %s" 
                         % (oracle_call, trn_loss, trn_acc, test_loss, test_acc))

            if self.predict_blinded:
                self.predict(sess=sess)

        return trn_losses, trn_accs, test_losses, test_accs

    def train(self):
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []

        if self.predict_blinded:
            sessions = ['BVSB']
        else:
            sessions = ['ENTROPY', 'BVSB', 'RANDOM']

        for session in sessions:
                trn_loss, trn_acc, test_loss, test_acc = self.single_tf_session(session_name=session)
                train_losses.append(trn_loss)
                test_losses.append(test_loss)
                train_accs.append(trn_acc)
                test_accs.append(test_acc)

        if not self.predict_blinded:
            plot(train_losses, test_losses, train_accs, test_accs)

    def predict(self, sess):
        softmax = sess.run(fetches=[self.output], feed_dict={self.X: self.blinded[:, 1:]})[0]
        labels = np.argmax(softmax, axis=1).astype(int)
        with open(self.blinded_path, 'w') as f:
            for i in range(labels.shape[0]):
                l = str(self.blinded[i][0]) + ", " + self.id_to_label_map[labels[i]] + '\n'
                f.write(l)


def plot(train_losses, test_losses, train_accs, test_accs):
    matplotlib.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 7))
    x = np.arange(1, 2501)
    plt.plot(x, test_losses[0], label='Entropy test', linewidth=2)
    plt.plot(x, test_losses[1], label='BVSB test', linewidth=2)
    plt.plot(x, test_losses[2], label='RL test', linewidth=2)
    plt.xlabel('Calls to Oracle')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Cross Entropy Loss Comparisons')
    plt.legend()
    plt.show()

    matplotlib.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 7))
    x = np.arange(1, 2501)
    plt.plot(x, test_accs[0], label='Entropy test', linewidth=2)
    plt.plot(x, test_accs[1], label='BVSB test', linewidth=2)
    plt.plot(x, test_accs[2], label='RL test', linewidth=2)
    plt.xlabel('Calls to Oracle')
    plt.ylabel('Classification Accuracy')
    plt.title('Classfication Accuracy Comparisons')
    plt.legend()
    plt.show()


def get_data(csv_path, labels_to_id_map):
    x = pd.read_csv(csv_path, header=None)
    x_arr = np.array(x.values)
    np.random.shuffle(x_arr)
    label_idx = x_arr.shape[1] - 1
    for i in range(x_arr.shape[0]):
        x_arr[i, label_idx] = labels_to_id_map[x_arr[i, label_idx]]
    one_hot_labels = get_one_hot_encoding(true_labels=x_arr[:, -1],  class_count=len(labels_to_id_map))
    return x_arr[:, :-1], one_hot_labels


def get_one_hot_encoding(true_labels, class_count):
    true_labels = [int(i) for i in true_labels]
    one_hot_matrix = np.zeros((len(true_labels), class_count), dtype=int)
    one_hot_matrix[np.arange(len(true_labels)), true_labels] = 1
    return one_hot_matrix


if __name__ == '__main__':
    # get the data
    labels_to_id_map = {'Endosomes': 0,
                        'Lysosome': 1,
                        'Mitochondria': 2,
                        'Peroxisomes': 3,
                        'Actin': 4,
                        'Plasma_Membrane': 5,
                        'Microtubules': 6,
                        'Endoplasmic_Reticulum': 7}
    id_to_labels_map = {}
    for k, v in labels_to_id_map.items():
        id_to_labels_map[v] = k

    # data
    easy_xtrain, easy_ytrain = get_data('Data/EASY_TRAIN.csv', labels_to_id_map)
    easy_xtest, easy_ytest = get_data('Data/EASY_TEST.csv', labels_to_id_map)
    mod_xtrain, mod_ytrain = get_data('Data/MODERATE_TRAIN.csv', labels_to_id_map)
    mod_xtest, mod_ytest = get_data('Data/MODERATE_TEST.csv', labels_to_id_map)
    diff_xtrain, diff_ytrain = get_data('Data/DIFFICULT_TRAIN.csv', labels_to_id_map)
    diff_xtest, diff_ytest = get_data('Data/DIFFICULT_TEST.csv', labels_to_id_map)

    # blinded data
    easy_blinded = np.array(pd.read_csv('Data/EASY_BLINDED.csv', header=None))
    mod_blinded = np.array(pd.read_csv('Data/MODERATE_BLINDED.csv', header=None))
    diff_blinded = np.array(pd.read_csv('Data/DIFFICULT_BLINDED.csv', header=None))

    # create a nn model that runs all 3 algorithms (BVSB, ENTROPY, RANDOM) on
    # one of the 3 data-sets
    # to run see the performance of all 3 algorithms, set predict_blinded to False
    # to make blinded predictions, set predict_blinded to True

    level = 'easy'  # can be 'easy', 'medium' or 'diff'
    if level == 'easy':
        nn = NN(m=26,
                num_classes=8,
                h1=60, h2=30,
                lrn_rate=0.05,
                dropout=0.5,
                weight_decay=0.001,
                seed=2017,
                x_train=easy_xtrain,
                y_train=easy_ytrain,
                x_test=easy_xtest,
                y_test=easy_ytest,
                blinded=easy_blinded,
                blinded_path='EASY_BLINDED_PRED.csv',
                predict_blinded=False,
                id_to_labels_map=id_to_labels_map)

    elif level == 'medium':
        nn = NN(m=26,
                num_classes=8,
                h1=80, h2=40,
                lrn_rate=0.05,
                dropout=0.5,
                weight_decay=0.001,
                seed=2017,
                x_train=mod_xtrain,
                y_train=mod_ytrain,
                x_test=mod_xtest,
                y_test=mod_ytest,
                predict_blinded=False,
                blinded=mod_blinded,
                blinded_path='MODERATE_BLINDED_PRED.csv',
                id_to_labels_map=id_to_labels_map)

    elif level == 'diff':
        nn = NN(m=52,
                num_classes=8,
                h1=65, h2=35,
                lrn_rate=0.05,
                dropout=0.5,
                weight_decay=0.001,
                seed=2017,
                x_train=diff_xtrain,
                y_train=diff_ytrain,
                x_test=diff_xtest,
                y_test=diff_ytest,
                predict_blinded=False,
                blinded=diff_blinded,
                blinded_path='DIFFICULT_BLINDED_PRED.csv',
                id_to_labels_map=id_to_labels_map)

    
