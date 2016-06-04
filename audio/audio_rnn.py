import json
import sys
import math

__author__ = 'anushabala'
from data_utils import get_audio_data, get_dataset
from config import *
import tensorflow as tf

import numpy as np
from datetime import datetime
from argparse import ArgumentParser


def load_data(audio_config, data_config):
    return get_dataset(data_config, audio_config)


def compute_lengths(config, X):
    num_examples = min(config.batch_size, X.shape[0])
    seq_lengths = np.zeros(shape=(num_examples))
    for i in xrange(0, num_examples):
        for j in xrange(0, config.max_seq_len):
            x_t = X[i][j]
            if not np.equal(x_t, -1 * np.ones_like(x_t)).all():
                seq_lengths[i] = j + 1
            else:
                break
    return seq_lengths


def update_stats(predictions, true_labels, stats_map):
    # print predictions, true_labels
    for (pred, true) in zip(predictions, true_labels):
        if true == 0 and pred == 0:
            stats_map["tn"] += 1
        if true == 0 and pred == 1:
            stats_map["fp"] += 1
        if true == 1 and pred == 0:
            stats_map["fn"] += 1
        if true == 1 and pred == 1:
            stats_map["tp"] += 1

    # print stats_map

def get_precision_recall(stats_map):
    if (stats_map["tp"] + stats_map["fp"]) > 0:
        precision = stats_map["tp"]/(stats_map["tp"] + stats_map["fp"])
    else:
        precision = 0.0
    if (stats_map["tp"] + stats_map["fn"]) > 0:
        recall = stats_map["tp"]/(stats_map["tp"] + stats_map["fn"])
    else:
        recall = 0.0

    return precision, recall


class AudioModel(object):
    def __init__(self, data_config=DataConfig(), model_config=ModelConfig()):
        self.config = model_config
        start_time = datetime.now()
        if self.config.verbose:
            print "\tLoading datasets..."

        self.X1_all, self.X2_all, self.labels_all = load_data(self.config, data_config)
        print self.X1_all.shape
        num_examples = self.X1_all.shape[0]
        indexes = np.arange(num_examples)
        np.random.shuffle(indexes)

        num_train = int(num_examples * data_config.train_split)
        train_idxes = indexes[0:num_train]
        val_idxes = indexes[num_train + 1:]

        self.X1_train = self.X1_all[train_idxes]
        self.X2_train = self.X2_all[train_idxes]
        self.labels_train = self.labels_all[train_idxes]

        self.X1_val = self.X1_all[val_idxes]
        self.X2_val = self.X2_all[val_idxes]
        self.labels_val = self.labels_all[val_idxes]

        if self.config.verbose:
            print "\tFinished loading dataset. Training data shape: ", self.X1_train.shape
            print "\tTest data shape: ", self.X1_val.shape
            print "\tTime taken: {}".format(str(datetime.now() - start_time))

        if self.config.verbose:
            print "\tAdding ops..."
        self.add_placeholders()

        conv_out1 = self.X1_batch
        conv_out2 = self.X2_batch
        self.conv_seq_length = self.config.max_seq_len
        self.conv_output_dimension = self.config.input_dimension
        if min(self.config.filter_heights) > 0 and min(self.config.filter_widths) > 0:
            conv_out1, conv_out2 = self.add_conv_layers(self.X1_batch, self.X2_batch)

        inputs1, inputs2 = self.reshape_batch(conv_out1, conv_out2)
        self.final_states_1, self.final_states_2 = self.add_model(inputs1, inputs2)
        self.output = self.add_fc_layer(self.final_states_1, self.final_states_2)
        self.loss = self.add_loss_op(self.output)
        self.predictions = self.compute_predictions(self.output)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        self.X1_batch = tf.placeholder(dtype='float32',
                                       shape=(None,
                                              self.config.max_seq_len,
                                              self.config.input_dimension),
                                       name='X1')
        self.X2_batch = tf.placeholder(dtype='float32',
                                       shape=(None,
                                              self.config.max_seq_len,
                                              self.config.input_dimension),
                                       name='X2')
        self.y_batch = tf.placeholder(dtype='int64',
                                      shape=(None,),
                                      name='labels')

        self.seq_lengths_X1 = tf.placeholder(dtype='int32',
                                             shape=(None,),
                                             name='seq_lengths_X1')

        self.seq_lengths_X2 = tf.placeholder(dtype='int32',
                                             shape=(None,),
                                             name='seq_lengths_X2')
        self.batch_size = tf.placeholder(dtype='int32')

    def add_conv_layers(self, inputs1, inputs2):
        inputs1 = tf.expand_dims(inputs1, -1)
        inputs2 = tf.expand_dims(inputs2, -1)
        filters = zip(self.config.filter_heights, self.config.filter_widths)

        seq_length = self.config.max_seq_len
        out_dim = self.config.input_dimension
        conv_out_length = seq_length
        conv_out_dim = out_dim
        conv_out1 = inputs1
        conv_out2 = inputs2
        # self.conv_seq_length = self.config.max_seq_len - self.config.filter_height + 1
        #     self.input_dimension = self.config.input_dimension - self.config.filter_width + 1

        for idx, (height, width) in enumerate(filters):
            filter_size = (height, width, 1, 1)
            with tf.variable_scope('ConvLayer'):
                filter = tf.get_variable('ConvFilter%d' % idx, shape=filter_size,
                                         initializer=tf.truncated_normal_initializer())

                conv_out1 = tf.nn.conv2d(inputs1, filter, strides=[1, 1, 1, 1], padding="VALID")
                conv_out2 = tf.nn.conv2d(inputs2, filter, strides=[1, 1, 1, 1], padding="VALID")

            conv_out_length = seq_length - height + 1
            conv_out_dim = out_dim - width + 1

            seq_length = conv_out_length
            out_dim = conv_out_dim
            inputs1 = conv_out1
            inputs2 = conv_out2

        self.conv_seq_length = conv_out_length
        self.conv_output_dimension = conv_out_dim
        conv_out1 = tf.squeeze(conv_out1, squeeze_dims=[3])
        conv_out2 = tf.squeeze(conv_out2, squeeze_dims=[3])
        print "Shape after convolutional layer:", conv_out1

        return conv_out1, conv_out2

    def reshape_batch(self, inp1, inp2):
        inputs1 = tf.split(split_dim=1, num_split=self.conv_seq_length, value=inp1)
        inputs2 = tf.split(split_dim=1, num_split=self.conv_seq_length, value=inp2)
        inputs1 = [tf.squeeze(x, squeeze_dims=[1]) for x in inputs1]
        inputs2 = [tf.squeeze(x, squeeze_dims=[1]) for x in inputs2]
        if self.config.verbose:
            print "Shape of data after split/squeeze ops:", inputs1
        return inputs1, inputs2

    def add_model(self, inputs1, inputs2):
        with tf.variable_scope("LSTMLayer"):
            self.lstm = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size / 2, input_size=self.conv_output_dimension,
                                                initializer=tf.truncated_normal_initializer())

        current_states_1 = self.lstm.zero_state(self.batch_size, dtype='float32')
        print current_states_1
        current_states_2 = self.lstm.zero_state(self.batch_size, dtype='float32')
        with tf.variable_scope("LSTMLayer"):
            _, final_states_1 = tf.nn.rnn(self.lstm, inputs1, initial_state=current_states_1)

        with tf.variable_scope("LSTMLayer", reuse=True):
            _, final_states_2 = tf.nn.rnn(self.lstm, inputs2, initial_state=current_states_2)


        if self.config.verbose:
            print "Completed add_model setup. Final states:", final_states_1

        return final_states_1, final_states_2

    def add_fc_layer(self, final_states_1, final_states_2):
        with tf.variable_scope('fc_layer'):
            # todo maybe try uniform_unit_scaling_initializer?
            if self.config.combine_type == 'diff':
                # difference betweeen states
                W = tf.get_variable('W', shape=(self.config.hidden_size, self.config.num_labels),
                                    initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
                combined_states = tf.abs(final_states_1 - final_states_2)  # absolute difference betwen the two states
                if self.config.normalize:
                    combined_states = tf.nn.l2_normalize(combined_states, dim=1)
            else:
                # default to concatenating
                W = tf.get_variable('W', shape=(self.config.hidden_size * 2, self.config.num_labels),
                                    initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
                combined_states = tf.concat(1, [final_states_1, final_states_2])

            b = tf.get_variable('b', shape=(self.config.num_labels,),
                                initializer=tf.truncated_normal_initializer(), dtype=tf.float32)

            output = tf.matmul(combined_states, W) + b

        return output

    def add_loss_op(self, output):
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output, self.y_batch))
        with tf.variable_scope('fc_layer', reuse=True):
            W = tf.get_variable('W')
            loss += self.config.reg / 2.0 * tf.reduce_sum(tf.pow(W, 2))
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)
        return optimizer.minimize(loss)

    def compute_predictions(self, output):
        sm = tf.nn.softmax(output)
        return tf.argmax(sm, 1)

    def evaluate(self, session, X1, X2, labels, summary_stats):
        # X1 = self.X1_val
        # X2 = self.X2_val
        # labels = self.labels_val
        num_examples = np.shape(X1)[0]
        batch_size = min(self.config.batch_size, num_examples)
        num_iters = int(math.ceil(float(num_examples) / batch_size))

        shuffled_idxes = np.arange(num_examples)
        np.random.shuffle(shuffled_idxes)
        val_loss = 0.0
        total_correct = 0

        for i in xrange(num_iters):
            this_batch_size = batch_size
            upper_bound = batch_size * (i + 1)
            if upper_bound > shuffled_idxes.shape[0]:
                this_batch_size = shuffled_idxes.shape[0] - batch_size * i
                upper_bound = shuffled_idxes.shape[0]
            shuffled_X1 = X1[shuffled_idxes[batch_size * i: upper_bound]]
            shuffled_X2 = X2[shuffled_idxes[batch_size * i: upper_bound]]
            shuffled_y = labels[shuffled_idxes[batch_size * i: upper_bound]]
            # seq_lengths_X1 = compute_lengths(self.config, shuffled_X1)
            # seq_lengths_X2 = compute_lengths(self.config, shuffled_X2)

            feed_dict = {self.X1_batch: shuffled_X1,
                         self.X2_batch: shuffled_X2,
                         self.y_batch: shuffled_y,
                         # self.seq_lengths_X1: seq_lengths_X1,
                         # self.seq_lengths_X2: seq_lengths_X2,
                         self.batch_size: this_batch_size}

            loss, predictions = session.run([self.loss, self.predictions], feed_dict=feed_dict)
            val_loss += loss

            assert len(predictions) == this_batch_size
            num_correct = np.sum(predictions == shuffled_y)
            update_stats(predictions, shuffled_y, summary_stats)

            total_correct += num_correct

        val_loss /= num_examples
        accuracy = float(total_correct) / num_examples

        return val_loss, accuracy

    def run_epoch(self, session, X1, X2, labels, summary_stats, print_every=5):
        num_examples = np.shape(X1)[0]

        num_iters = int(math.ceil(float(num_examples) / self.config.batch_size))

        shuffled_idxes = np.arange(num_examples)
        np.random.shuffle(shuffled_idxes)
        epoch_loss = 0.0
        total_correct = 0
        for i in xrange(num_iters):
            batch_size = self.config.batch_size
            upper_bound = batch_size * (i + 1)
            if upper_bound > shuffled_idxes.shape[0]:
                batch_size = shuffled_idxes.shape[0] - (self.config.batch_size * i)
                upper_bound = shuffled_idxes.shape[0]
            shuffled_X1 = X1[shuffled_idxes[self.config.batch_size * i: upper_bound]]
            shuffled_X2 = X2[shuffled_idxes[self.config.batch_size * i: upper_bound]]
            shuffled_y = labels[shuffled_idxes[self.config.batch_size * i: upper_bound]]
            # seq_lengths_X1 = compute_lengths(self.config, shuffled_X1)
            # print "Computing sequence lengths for X1"
            # print shuffled_X1
            # print seq_lengths_X1
            # seq_lengths_X2 = compute_lengths(self.config, shuffled_X2)

            # print "Computing sequence lengths for X2"
            # print shuffled_X2
            # print seq_lengths_X2

            feed_dict = {self.X1_batch: shuffled_X1,
                         self.X2_batch: shuffled_X2,
                         self.y_batch: shuffled_y,
                         # self.seq_lengths_X1: seq_lengths_X1,
                         # self.seq_lengths_X2: seq_lengths_X2,
                         self.batch_size: batch_size}
            loss, predictions, _ = session.run([self.loss, self.predictions, self.train_op], feed_dict=feed_dict)
            epoch_loss += loss

            assert len(predictions) == batch_size
            num_correct = np.sum(predictions == shuffled_y)
            total_correct += num_correct
            update_stats(predictions, shuffled_y, summary_stats)

            if i % print_every == 0:
                sys.stdout.write("\r\titer (%d/%d):\tLoss: %2.4f\tTrain accuracy:%2.2f" % (
                    i + 1, num_iters, loss / batch_size, float(num_correct) / batch_size))
                sys.stdout.flush()

        epoch_loss /= num_examples
        sys.stdout.write("\n")
        return epoch_loss, total_correct


def run(data_config, model_config):
    stats_file = data_config.stats_file
    checkpoint_name = data_config.params_file
    eval_info = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "data_params": data_config.to_json(),
                 "model_params": model_config.to_json(), "train_stats":[], "val_stats":[]}
    start_setup = datetime.now()
    if config.verbose:
        print "Starting model setup.."
    model = AudioModel(data_config, model_config)
    finish_setup = datetime.now()
    if config.verbose:
        print "Finished setup. Time taken: {}".format(str(finish_setup - start_setup))
    init = tf.initialize_all_variables()

    train_start = datetime.now()

    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(init)

        # evaluate train and val data once before training
        print "Before training:"
        epoch_train_stats = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
        init_train_loss, init_train_acc = model.evaluate(session, model.X1_train, model.X2_train, model.labels_train, summary_stats=epoch_train_stats)
        eval_info["train_loss"].append((0, init_train_loss))
        eval_info["train_acc"].append((0, init_train_acc))
        eval_info["train_stats"].append((0, epoch_train_stats))
        train_prec, train_recall = get_precision_recall(epoch_train_stats)
        print "\tTraining loss: %2.4f\tAccuracy: %2.4f\tPrecision: %2.4f\tRecall: %2.4f" % (init_train_loss, init_train_acc, train_prec, train_recall)

        val_stats = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
        init_val_loss, init_val_acc = model.evaluate(session, model.X1_val, model.X2_val, model.labels_val, summary_stats=val_stats)
        eval_info["val_loss"].append((0, init_val_loss))
        eval_info["val_acc"].append((0, init_val_acc))
        eval_info["val_stats"].append((0, val_stats))
        val_prec, val_recall = get_precision_recall(val_stats)
        print "\tValidation loss: %2.4f\tAccuracy: %2.4f\tPrecision: %2.4f\tRecall: %2.4f" % (init_val_loss, init_val_acc, val_prec, val_recall)

        if config.verbose:
            print "Started model training.."

        best_val_loss = init_val_loss

        for epoch in xrange(model_config.num_epochs):
            epoch_train_stats = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}

            num_examples = model.X1_train.shape[0]
            # epoch_loss, train_acc = \
            model.run_epoch(session, model.X1_train, model.X2_train, model.labels_train,
                                                        summary_stats=epoch_train_stats,
                                                        print_every=config.print_every)
            # train_acc = float(total_correct) / num_examples
            epoch_loss, train_acc = model.evaluate(session, model.X1_train, model.X2_train, model.labels_train, summary_stats=epoch_train_stats)
            eval_info["train_loss"].append((epoch, epoch_loss))
            eval_info["train_acc"].append((epoch, train_acc))
            eval_info["train_stats"].append((epoch, epoch_train_stats))
            train_prec, train_recall = get_precision_recall(epoch_train_stats)

            if epoch % config.anneal_every:
                config.lr = config.lr * config.anneal_by

            if epoch % config.evaluate_every == 0:
                val_stats = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
                val_loss, val_accuracy = model.evaluate(session, model.X1_val, model.X2_val, model.labels_val, summary_stats=val_stats)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    saver.save(session, checkpoint_name+'.best')
                eval_info["val_loss"].append((epoch, val_loss))
                eval_info["val_acc"].append((epoch, val_accuracy))
                eval_info["val_stats"].append((epoch,val_stats))
                val_prec, val_recall = get_precision_recall(val_stats)
                print "\tValidation loss: %2.4f\tValidation accuracy: %2.4f\tPrecision: %2.4f\tRecall:%2.4f" % (
                    val_loss, val_accuracy, val_prec, val_recall)

            if epoch % data_config.save_every == 0:
                # save model and loss/acc histories so we don't lose information when an early stop occurs
                saver.save(session, checkpoint_name, global_step=epoch)
                json.dump(eval_info, open(stats_file, 'w'))

            print "Epoch (%d/%d) completed:\tTotal loss: %2.4f\tTrain accuracy: %2.4f\tPrecision: %2.4f\tRecall: %2.4f\tTime: %s" % (
                epoch + 1, config.num_epochs,
                epoch_loss,
                train_acc,
                train_prec,
                train_recall,
                str(datetime.now() - train_start))

    print "Finished training. Time taken: {}".format(datetime.now() - train_start)

    json.dump(eval_info, open(stats_file, 'w'))


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
    parser.add_argument('--filter_heights', nargs='+', default=[11], type=int,
                        help='Height of filter for convolutional layer. '
                             'If 0 or less than 0, no convolutional layer is applied. '
                             'Specify one value for each convolutional layer.')
    parser.add_argument('--filter_widths', type=int, nargs='+', default=[1],
                        help='Width of filter for convolutional layer. '
                             'If 0 or less than 0, no convolutional layer is applied. '
                             'Specify one value for each convolutional layer')
    # parser.add_argument('')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use_fft', action='store_true')
    parser.add_argument('--time_block', type=float, default=1.0)
    parser.add_argument('--print_every', type=int, default=5, help='# of iterations after which to print status')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='# of epochs after which to evaluate on validation data')
    parser.add_argument('--train_split', type=float, default=0.8, help='% of data to use as training data')
    parser.add_argument('--anneal_by', type=float, default=0.95, help='Amount to anneal learning rate by')
    parser.add_argument('--anneal_every', type=int, default=1, help='# of epochs to anneal after')
    parser.add_argument('--stats_file', type=str, default='../eval_stats.json', help='file to log stats to')
    parser.add_argument('--combine_type', type=str, choices=['concat', 'diff'], default='concat',
                        help='How to combine final hidden states.')
    parser.add_argument('--checkpoints_name', type=str, default='params/audio-model')
    parser.add_argument('--save_every', type=int, default=1, help='Number of epochs after which to save params')
    parser.add_argument('--normalize', action='store_true', help='Normalize differences before passing to fc')

    args = parser.parse_args()
    data_config = DataConfig(max_examples=args.max_examples,
                             pickle_file=args.pickle_file,
                             mp3_dir=args.mp3_dir,
                             wav_dir=args.wav_dir,
                             max_files=args.max_files,
                             mappings_file=args.mappings_file,
                             save_data=args.save_data,
                             reload_data=args.reload_data,
                             stats_file=args.stats_file,
                             train_split=args.train_split,
                             params_file=args.checkpoints_name,
                             save_every=args.save_every)
    config = ModelConfig(learning_rate=args.lr,
                         reg=args.reg,
                         max_audio_len=args.max_audio_len,
                         batch_size=args.batch_size,
                         num_epochs=args.epochs,
                         verbose=args.verbose,
                         hidden_size=args.hidden_size,
                         print_every=args.print_every,
                         evaluate_every=args.evaluate_every,
                         anneal_by=args.anneal_by,
                         anneal_every=args.anneal_every,
                         use_fft=args.use_fft,
                         time_block=args.time_block,
                         filter_height=args.filter_heights,
                         filter_width=args.filter_widths,
                         combine_type=args.combine_type,
                         normalize=args.normalize)

    run(data_config, config)
