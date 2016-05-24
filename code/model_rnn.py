import tensorflow as tf
from get_data import get_data
from model import LanguageModel
import math
from utils import Vocab
import numpy as np


class Config:
    n_class = 5  # ?
    batch_size = 1
    embed_size = 10
    window_size = -1  # ?
    hidden_size = 5
    reg = 0
    max_epochs = 1
    lr = 0
    max_steps = -1


class Model_RNN(LanguageModel):
    def load_data(self):
        pair_fname = '../lastfm_train_mappings.txt'
        lyrics_path = '../lyrics/data/lyrics/train/'
        X_train, y_train, self.word_counts, self.config.max_steps = get_data(pair_fname, lyrics_path, threshold=100,
                                                                             n_class=self.config.n_class)

        self.vocab = Vocab()
        self.vocab.construct(self.word_counts.keys())
        self.encoded_train_1 = np.zeros((len(X_train), self.config.max_steps))  # need to handle this better.
        self.encoded_train_2 = np.zeros((len(X_train), self.config.max_steps))
        for i in range(len(X_train)):
            self.encoded_train_1[i, :len(X_train[i][0])] = [self.vocab.encode(word) for word in X_train[i][0]]
            self.encoded_train_2[i, :len(X_train[i][1])] = [self.vocab.encode(word) for word in X_train[i][1]]

    def add_placeholders(self):
        self.X1 = tf.placeholder(tf.int32, shape=(None, self.config.max_steps), name='X1')
        self.X2 = tf.placeholder(tf.int32, shape=(None, self.config.max_steps), name='X2')
        self.y = tf.placeholder(tf.int32, shape=(None, self.config.max_steps), name='y')
        # self.initial_state = tf.placeholder(tf.float32, shape=(None, self.config.hidden_size), name='initial_state')
        self.seq_len = tf.placeholder(tf.int32, shape=(None), name='seq_len')  # for variable length sequences

    def add_embedding(self):
        L = tf.get_variable('L', shape=(len(self.word_counts.keys()), self.config.embed_size))
        inputs1 = tf.nn.embedding_lookup(L, self.X1)  # batch_size x
        inputs2 = tf.nn.embedding_lookup(L, self.X2)
        inputs1 = tf.split(1, self.config.max_steps, inputs1)
        inputs1 = [tf.squeeze(x) for x in inputs1]
        inputs2 = tf.split(1, self.config.max_steps, inputs2)
        inputs2 = [tf.squeeze(x) for x in inputs2]
        return inputs1, inputs2

    def add_model(self, inputs1, inputs2, seq_len):
        # self.initial_state = tf.constant(np.zeros(()), dtype=tf.float32)
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)  # RNN
        self.initial_state = self.lstm.zero_state(self.config.batch_size, tf.float32)
        rnn_outputs1, _ = tf.nn.rnn(self.lstm, inputs1, initial_state=self.initial_state, sequence_length=seq_len)
        rnn_outputs2, _ = tf.nn.rnn(self.lstm, inputs2, initial_state=self.initial_state, sequence_length=seq_len)
        rnn_outputs1 = [x[-1] for x in outputs1]
        rnn_outputs2 = [x[-1] for x in outputs2]
        rnn_nutputs = tf.concat(1, [rnn_outputs1, rnn_output2])

        return rnn_outputs

    def add_final_projection(self, rnn_outputs):
        # rnn_outputs is a list of length batch_size of lengths = seq_len. Where each list element is ??. I think.
        Whc = tf.get_variable('Whc', shape=(2 * self.config.hidden_size, self.config.n_class))
        bhc = tf.get_variable('bhc', shape=(self.config.n_class,))
        final_outputs = [tf.matmul(x[-1], Whc) + bhc for x in rnn_outputs]
        return final_outputs

    def add_loss(self, outputs):
        loss = tf.sequence_loss([output], [self.y], tf.ones_like(self.y, dtype=tf.float32))
        return loss

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        return train_op

    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.inputs1, self.inputs2 = self.add_embedding()
        self.rnn_outputs = self.add_model(self.inputs1, self.inputs2, self.seq_len)
        self.final_outputs = self.add_final_projection()
        self.predictions = [tf.nn.softmax(tf.cast(x, 'float64')) for x in
                            self.final_outputs]  # to avoid numerical errors
        self.calculate_loss = self.add_loss_op(outputs)
        self.train_step = self.add_training_op(self.calculate_loss)

    def run_epoch(self, session, X, y):  # X and y are 2D np arrays
        config = self.config
        state = tf.zeros([batch_size, lstm.state_size])
        data_len = np.shape(X)[0]
        index = np.arange(data_len)
        np.random.shuffle(index)
        n_batches = data_len // self.config.batch_size

        loss = 0.0
        for batch_num in range(n_batches):

            X_batch = X[index[batch_num * batch_size: (batch_num + 1) * batch_size], :]
            y_batch = y[index[batch_num * batch_size: (batch_num + 1) * batch_size], :]
            state = tf.constant(np.zeros(len(X_batch, self.config.hidden_size)), dtype=tf.float32)

            feed_dict = {self.X1: x_batch[:, 0],
                         self.X2: x_batch[:, 1],
                         self.y: y_batch,
                         self.initial_state: state}

            loss, state, _ = session.run([self.loss, self.final_state, train_op], feed_dict=feed_dict)
            total_loss.append(loss)

            if verbose and (batch_num + 1) % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(batch_num + 1, n_batches, np.exp(np.mean(total_loss))))
                sys.stdout.flush()
            if verbose:
                sys.stdout.write('\r')

            return np.exp(np.mean(total_loss))


def run():
    config = Config()

    # We create the training model and generative model
    with tf.variable_scope('Model_RNN') as scope:
        model = Model_RNN(config)
        print 'asdf'
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
    exit()
    with tf.Session() as session:
        best_val_pp = float('inf')
        best_val_epoch = 0

        session.run(init)
        for epoch in xrange(config.max_epochs):
            print 'Epoch {}'.format(epoch)
            start = time.time()
            ###
            train_pp = model.run_epoch(
                session, model.encoded_train,
                train_op=model.train_step)
            # valid_pp = model.run_epoch(session, model.encoded_valid)
            print 'Training perplexity: {}'.format(train_pp)
            # print 'Validation perplexity: {}'.format(valid_pp)
            # if valid_pp < best_val_pp:
            # best_val_pp = valid_pp
            # best_val_epoch = epoch
            # saver.save(session, './ptb_rnnlm.weights')
            # if epoch - best_val_epoch > config.early_stopping:
            # break
            print 'Total time: {}'.format(time.time() - start)


if __name__ == '__main__':
    run()
