import tensorflow as tf
import numpy as np
from tqdm import tqdm 

class MultiTaskModel:

    def __init__(self, labels, feature_size, shared_layers=3, individual_layers=2):
        self.labels = labels
        self.feature_size = feature_size
        self.batch_size = 256

        self.shared_size = 128
        self.shared_layers = 2
        self.indiv_size = 128
        self.indiv_layers = 1

        self._g_acc = None
        self._g_opt = None
        self._g_predict = None
        self._build_graph()


    def _build_graph(self):

        features = tf.placeholder(shape=[None, self.feature_size], name="features", dtype=tf.float32)
        labels = tf.placeholder(shape=[None, self.labels], name="labels", dtype=tf.float32)

        with tf.name_scope('shared') as scope:
            shared_layer = tf.contrib.layers.fully_connected(features, self.shared_size)
            
            for _ in range(self.shared_layers-1):
                shared_layer = tf.contrib.layers.fully_connected(shared_layer, self.shared_size)
        
        split_inputs = []
        for l in range(self.labels + 1):
            nodes = self.shared_size // self.labels
            s = shared_layer[ :, l*nodes: (l+1)*nodes]
            split_inputs.append(s)

        with tf.name_scope('individual') as scope:
            indiv_nodes = [tf.contrib.layers.fully_connected(s, self.indiv_size) for s in split_inputs]

            final_logits = []
            for i, n in enumerate(indiv_nodes):
                output = None
                for _ in range(self.indiv_layers-1):
                    n = tf.contrib.layers.fully_connected(n, self.indiv_size)
                    output = tf.contrib.layers.fully_connected(n, 1)
                if output is None:
                   final_logits = indiv_nodes

        logits = tf.concat(final_logits, axis=1)
        logits = tf.contrib.layers.fully_connected(logits, 4)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
        opt = tf.train.AdamOptimizer(0.0001).minimize(loss)

        predict = tf.nn.softmax(logits)
        equal = tf.equal(tf.argmax(predict, axis=1), tf.argmax(labels, axis=1))
        acc = tf.reduce_mean(tf.cast(equal, dtype=tf.int32))

        self._g_features = features
        self._g_labels = labels

        self._g_acc = acc
        self._g_opt = opt
        self._g_loss = loss
        self._g_predict = predict

    def train(self, session, epochs, train_tuple, test_tuple):
        
        for i in range(epochs):

            losses = []
            for features, labels in self._to_batch(*train_tuple):
                ops = [self._g_opt, self._g_loss]
                feed_dict = {self._g_features: features, self._g_labels:labels}
                
                _, loss = session.run(ops, feed_dict=feed_dict)
                losses.append(loss)
            print("Train Loss:", np.mean(losses))

            train_loss, train_acc = [], []
            for features, labels in self._to_batch(*train_tuple):
                ops = [self._g_loss, self._g_acc]
                feed_dict = {self._g_features: features, self._g_labels:labels}
                
                loss, acc = session.run(ops, feed_dict=feed_dict)
                train_loss.append(loss)
                train_acc.append(acc)
            print("Train Loss: {}, Train Acc: {}".format(np.mean(train_loss), np.mean(train_acc)))

            test_loss, test_acc = [], []
            for features, labels in self._to_batch(*test_tuple):
                ops = [self._g_loss, self._g_acc]
                feed_dict = {self._g_features: features, self._g_labels:labels}
                
                loss, acc = session.run(ops, feed_dict=feed_dict)
                test_loss.append(loss)
                test_acc.append(acc)
            print("Test Loss: {}, Test Acc: {}".format(np.mean(test_loss), np.mean(test_acc)))


    def _to_batch(self, features, labels, do_prog_bar=False):
        assert features.shape[0] == labels.shape[0]
        size = features.shape[0]
        for i in tqdm(range((size//self.batch_size)+1), leave=do_prog_bar):
            train_batch = features[i * self.batch_size: (i+1) * self.batch_size]
            labels_batch = labels[i * self.batch_size: (i+1) * self.batch_size]
            yield train_batch, labels_batch




def test_with_mnist(nn):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    train = (mnist.train.images, mnist.train.labels)
    test = (mnist.test.images, mnist.test.labels)

    cols = [4,5,6,7,8,9] 

    train_numbers = np.argmax(mnist.train.labels, axis=1)
    delete_rows = [i for i, n in enumerate(train_numbers) if n > 4]
    one_to_four_train_feat = np.delete(train[0], delete_rows, axis=0)
    _otf = np.delete(train[1], delete_rows, axis=0)
    one_to_four_train_lab = np.delete(_otf, cols, axis=1)

    test_numbers = np.argmax(mnist.test.labels, axis=1)
    delete_rows = [i for i, n in enumerate(test_numbers) if n > 4]
    one_to_four_test_feat = np.delete(test[0], delete_rows, axis=0)
    _otf = np.delete(test[1], delete_rows, axis=0)
    one_to_four_test_lab = np.delete(_otf, cols, axis=1)


    otf_train = (one_to_four_train_feat, one_to_four_train_lab)
    otf_test = (one_to_four_test_feat, one_to_four_test_lab)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nn.train(sess, 30, otf_train, otf_test)
        




    
if __name__ == "__main__":
    labels = 4
    features_size = 784
    nn = MultiTaskModel(labels, features_size)

    test_with_mnist(nn) 

  

