__author__ = 'anushabala'
from data_utils import get_audio_data, get_dataset
from config import *
import tensorflow as tf

import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from matplotlib import pyplot as plt


def load_data(audio_config, data_config):
    return get_dataset(data_config, audio_config)


def compute_lengths(config, X):
        seq_lengths = np.zeros(shape=(config.batch_size,))
        for i in xrange(0, config.batch_size):
            for j in xrange(0, config.max_seq_len):
                x_t = X[i][j]
                if not np.equal(x_t, -1 * np.ones_like(x_t)).all():
                    seq_lengths[i] = j+1
                else:
                    break
        return seq_lengths


class AudioModel(object):
    def __init__(self, data_config=DataConfig(), model_config=ModelConfig()):
        self.config = model_config
        start_time = datetime.now()
        if self.config.verbose:
            print "\tLoading datasets..."

        self.X1_train, self.X2_train, self.labels = load_data(self.config, data_config)
        if self.config.verbose:
            print "\tFinished loading dataset. Dataset shape: ", self.X1_train.shape
            print "\tTime taken: {}".format(str(datetime.now()-start_time))

        if self.config.verbose:
            print "\tAdding ops..."
        self.add_placeholders()
        inputs1, inputs2 = self.reshape_batch()
        self.initial_state = tf.constant(np.zeros((self.config.batch_size, self.config.hidden_size)), dtype=tf.float32)
        self.final_states_1, self.final_states_2 = self.add_model(inputs1, inputs2)
        self.output = self.add_fc_layer(self.final_states_1, self.final_states_2)
        self.loss = self.add_loss_op(self.output)
        self.predictions = self.compute_predictions(self.output)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        self.X1_batch = tf.placeholder(dtype='float32',
                                       shape=(self.config.batch_size,
                                              self.config.max_seq_len,
                                              self.config.input_dimension),
                                       name='X1')
        self.X2_batch = tf.placeholder(dtype='float32',
                                       shape=(self.config.batch_size,
                                              self.config.max_seq_len,
                                              self.config.input_dimension),
                                       name='X2')
        self.y_batch = tf.placeholder(dtype='int32',
                                      shape=(self.config.batch_size,),
                                      name='labels')

        self.seq_lengths_X1 = tf.placeholder(dtype='int32',
                                             shape=(self.config.batch_size,),
                                             name='seq_lengths_X1')

        self.seq_lengths_X2 = tf.placeholder(dtype='int32',
                                             shape=(self.config.batch_size,),
                                             name='seq_lengths_X2')

    def reshape_batch(self):
        inputs1 = tf.split(split_dim=1, num_split=self.config.max_seq_len, value=self.X1_batch)
        inputs2 = tf.split(split_dim=1, num_split=self.config.max_seq_len, value=self.X2_batch)
        inputs1 = [tf.squeeze(x, squeeze_dims=[1]) for x in inputs1]
        inputs2 = [tf.squeeze(x, squeeze_dims=[1]) for x in inputs2]
        # if self.config.verbose:
        #     print "Shape of data after split/squeeze ops:", inputs1
        return inputs1, inputs2

    def add_model(self, inputs1, inputs2):
        with tf.variable_scope("LSTMLayer"):
            self.lstm = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size/2, input_size=self.config.input_dimension, initializer=tf.truncated_normal_initializer())

        current_states_1 = self.lstm.zero_state(self.config.batch_size, dtype='float32')
        print current_states_1
        current_states_2 = self.lstm.zero_state(self.config.batch_size, dtype='float32')
        with tf.variable_scope("LSTMLayer"):
            _, final_states_1 = tf.nn.rnn(self.lstm, inputs1, initial_state=current_states_1,
                                          sequence_length=self.seq_lengths_X1)

        with tf.variable_scope("LSTMLayer", reuse=True):
            _, final_states_2 = tf.nn.rnn(self.lstm, inputs2, initial_state=current_states_2,
                                          sequence_length=self.seq_lengths_X2)

        # for i in xrange(0, self.config.max_seq_len):
        #     x1_t = inputs1[i]
        #     x2_t = inputs2[i]
        #     with tf.variable_scope("LSTMLayer", reuse=True):
        #         _, new_states_1 = self.lstm(x1_t, current_states_1)
        #         _, new_states_2 = self.lstm(x2_t, current_states_2)
        #
        #     for j in xrange(0, self.config.batch_size):
        #         # if audio sequence has already ended, don't update final hidden state any further
        #         if tf.cast(tf.not_equal(tf.reduce_sum(tf.cast(tf.not_equal(x1_t[j,:], 0.0), dtype='int32')), 0), dtype='int32') == 1:
        #             final_states_1[j] = new_states_1[j]
        #         if tf.cast(tf.not_equal(tf.reduce_sum(tf.cast(tf.not_equal(x2_t[j,:], 0.0), dtype='int32')), 0), dtype='int32') == 1:
        #             final_states_2[j] = new_states_2[j]
        #
        #     current_states_1 = new_states_1
        #     current_states_2 = new_states_2

        if self.config.verbose:
            print "Completed add_model setup. Final states:", final_states_1

        return final_states_1, final_states_2

    def add_fc_layer(self, final_states_1, final_states_2):
        with tf.variable_scope('fc_layer'):
            # todo maybe try uniform_unit_scaling_initializer?
            W = tf.get_variable('W', shape=(self.config.hidden_size * 2, self.config.num_labels),
                                initializer=tf.truncated_normal_initializer(), dtype=tf.float32)

            b = tf.get_variable('b', shape=(self.config.num_labels,),
                                initializer=tf.truncated_normal_initializer(), dtype=tf.float32)

            concat_states = tf.concat(1, [final_states_1, final_states_2])

            output = tf.matmul(concat_states, W) + b

        return output

    def add_loss_op(self, output):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output, self.y_batch))
        with tf.variable_scope('fc_layer', reuse=True):
            W = tf.get_variable('W')
            loss += self.config.reg / 2.0 * tf.reduce_sum(tf.pow(W, 2))
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.config.lr)
        return optimizer.minimize(loss)

    def compute_predictions(self, output):
        sm = tf.nn.softmax(output)
        return tf.argmax(sm, 1)

    def run_epoch(self, session, X1, X2, labels, print_every=5):
        num_examples = np.shape(X1)[0]
        num_iters = num_examples / self.config.batch_size
        batch_size = self.config.batch_size

        shuffled_idxes = np.arange(num_examples)
        np.random.shuffle(shuffled_idxes)
        epoch_loss = 0.0
        total_correct = 0
        for i in xrange(num_iters):
            shuffled_X1 = X1[shuffled_idxes[batch_size * i: batch_size * (i + 1)]]
            shuffled_X2 = X2[shuffled_idxes[batch_size * i: batch_size * (i + 1)]]
            shuffled_y = labels[shuffled_idxes[batch_size * i: batch_size * (i + 1)]]
            seq_lengths_X1 = compute_lengths(self.config, shuffled_X1)
            # print "Computing sequence lengths for X1"
            # print shuffled_X1
            # print seq_lengths_X1
            seq_lengths_X2 = compute_lengths(self.config, shuffled_X2)

            # print "Computing sequence lengths for X2"
            # print shuffled_X2
            # print seq_lengths_X2

            feed_dict = {self.X1_batch: shuffled_X1,
                         self.X2_batch: shuffled_X2,
                         self.y_batch: shuffled_y,
                         self.seq_lengths_X1: seq_lengths_X1,
                         self.seq_lengths_X2: seq_lengths_X2}

            loss, predictions, _ = session.run([self.loss, self.predictions, self.train_op], feed_dict=feed_dict)
            # print states1
            # print states2
            epoch_loss += loss
            # print predictions
            # print shuffled_y

            assert len(predictions) == batch_size
            num_correct = np.sum(predictions == shuffled_y)
            total_correct += num_correct
            if i % print_every == 0:
                print "\titer (%d/%d):\tLoss: %2.4f\tTrain accuracy:%2.2f" % (
                i+1, num_iters, loss, float(num_correct) / self.config.batch_size)

        epoch_loss /= num_iters
        return epoch_loss, total_correct


def run(data_config, model_config):
    start_setup = datetime.now()
    if config.verbose:
        print "Starting model setup.."
    model = AudioModel(data_config, model_config)
    finish_setup = datetime.now()
    if config.verbose:
        print "Finished setup. Time taken: {}".format(str(finish_setup - start_setup))
    init = tf.initialize_all_variables()

    train_start = datetime.now()
    if config.verbose:
        print "Started model training.."
    with tf.Session() as session:
        session.run(init)
        # todo load val data and check accuracy
        for epoch in xrange(model_config.num_epochs):
            num_examples = model.X1_train.shape[0]
            epoch_loss, total_correct = model.run_epoch(session, model.X1_train, model.X2_train, model.labels,
                                                        print_every=config.print_every)
            print "Epoch (%d/%d) completed:\tTotal loss: %2.4f\tTrain accuracy: %2.4f" % (epoch+1, config.num_epochs,
                                                                                          epoch_loss,
                                                                                          float(total_correct)/num_examples)
            print "Time elapsed since training start: {}".format(str(datetime.now() - train_start))

    print "Finished training. Time taken: {}".format(datetime.now() - train_start)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--mp3_dir', type=str, default='../data/audio/train')
    parser.add_argument('--wav_dir', type=str, default='../data/audio/train_wav')
    parser.add_argument('--reload_data', action='store_true')
    parser.add_argument('--pickle_file', type=str, default='../data/audio/train_data.pkl')
    parser.add_argument('--max_files', type=int, default=-1)
    parser.add_argument('--max_examples', type=int, default=-1)
    parser.add_argument('--mappings_file', type=str, default='../lastfm_train_mappings.txt')
    parser.add_argument('--save_data', action='store_true')
    parser.add_argument('--max_audio_len', type=int, default=15, help='Maximum audio length in seconds')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--reg', type=float, default=0)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--print_every', type=int, default=5, help='# of iterations after which to print status')

    args = parser.parse_args()

    data_config = DataConfig(max_examples=args.max_examples,
                             pickle_file=args.pickle_file,
                             mp3_dir=args.mp3_dir,
                             wav_dir=args.wav_dir,
                             max_files=args.max_files,
                             mappings_file=args.mappings_file,
                             save_data=args.save_data,
                             reload_data=args.reload_data)
    config = ModelConfig(learning_rate=args.lr,
                         reg=args.reg,
                         max_audio_len=args.max_audio_len,
                         batch_size=args.batch_size,
                         num_epochs=args.epochs,
                         verbose=args.verbose,
                         hidden_size=args.hidden_size,
                         print_every=args.print_every)

    run(data_config, config)
