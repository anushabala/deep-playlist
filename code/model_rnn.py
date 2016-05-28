import tensorflow as tf
from get_data import get_data
from model import LanguageModel
import math
from utils import Vocab
import numpy as np
import time
 
class Config:
    n_class     = 5 # ?
    batch_size  = 1
    embed_size  = 10
    window_size = -1 # ?
    hidden_size = 5
    reg         = 0
    max_epochs  = 1
    lr          = 0.1
    max_steps   = -1

class Model_RNN(LanguageModel):
    
    def load_data(self):
        pair_fname  = '../lastfm_train_mappings.txt'
        lyrics_path = '../lyrics/data/lyrics/train/'
    
        # X_train is a list of all examples. each examples is a 2-len list. each element is a list of words in lyrics.
        # word_counts is a dictionary that maps 
        X_train, l_train, self.word_counts, self.config.max_steps = get_data(pair_fname, lyrics_path, threshold=100, n_class=self.config.n_class)
        self.labels_train = np.zeros((len(X_train),self.config.n_class))
        self.labels_train[range(len(X_train)),l_train] = 1
    
        self.vocab = Vocab()
        self.vocab.construct(self.word_counts.keys())

        self.encoded_train_1 = np.zeros((len(X_train), self.config.max_steps)) # need to handle this better. 
        self.encoded_train_2 = np.zeros((len(X_train), self.config.max_steps))
        for i in range(len(X_train)):
            self.encoded_train_1[i,:len(X_train[i][0])] = [self.vocab.encode(word) for word in X_train[i][0]]       
            self.encoded_train_2[i,:len(X_train[i][1])] = [self.vocab.encode(word) for word in X_train[i][1]]       


    def add_placeholders(self):
        self.X1            = tf.placeholder(tf.int32,   shape=(None, self.config.max_steps), name='X1')
        self.X2            = tf.placeholder(tf.int32,   shape=(None, self.config.max_steps), name='X2')
        self.labels        = tf.placeholder(tf.float32,   shape=(None, self.config.n_class), name='labels')
        #self.initial_state = tf.placeholder(tf.float32, shape=(None, self.config.hidden_size), name='initial_state')
        self.seq_len1      = tf.placeholder(tf.int32,   shape=(None),                        name='seq_len1') # for variable length sequences
        self.seq_len2      = tf.placeholder(tf.int32,   shape=(None),                        name='seq_len2') # for variable length sequences

    def add_embedding(self):
        L = tf.get_variable('L', shape=(len(self.word_counts.keys()), self.config.embed_size), dtype=tf.float32) 
        inputs1 = tf.nn.embedding_lookup(L, self.X1) # self.X1 is batch_size x self.config.max_steps 
        inputs2 = tf.nn.embedding_lookup(L, self.X2) # input2 is batch_size x self.config.max_steps x self.config.embed_size
        inputs1 = tf.split(1, self.config.max_steps, inputs1) # list of len self.config.max_steps where each element is batch_size x self.config.embed_size
        inputs1 = [tf.squeeze(x) for x in inputs1]
        inputs2 = tf.split(1, self.config.max_steps, inputs2) # list of len self.config.max_steps where each element is batch_size x self.config.embed_size
        inputs2 = [tf.squeeze(x) for x in inputs2]
        print 'onh'
        print inputs1[0].get_shape
        return inputs1, inputs2

    def add_model(self, inputs1, inputs2, seq_len1, seq_len2):
        #self.initial_state = tf.constant(np.zeros(()), dtype=tf.float32)
        print 'adsf add_model'
        self.initial_state = tf.constant(np.zeros((self.config.batch_size,self.config.hidden_size)), dtype=tf.float32)
        rnn_outputs  = []
        rnn_outputs1 = []
        rnn_outputs2 = []
        h_curr1 = self.initial_state
        h_curr2 = self.initial_state
        print 'nthgnghn'
        with tf.variable_scope('rnn'):
            Whh = tf.get_variable('Whh', shape=(self.config.hidden_size,self.config.hidden_size), dtype=tf.float32)
            Wxh = tf.get_variable('Wxh', shape=(self.config.embed_size,self.config.hidden_size),  dtype=tf.float32)
            b1  = tf.get_variable('bhx', shape=(self.config.hidden_size,),                        dtype=tf.float32)
            print Wxh.get_shape
            print inputs1[0].get_shape
            print inputs2[0].get_shape
            for i in range(self.config.max_steps):
                h_curr2 = tf.matmul(h_curr2,Whh) 
                h_curr2 += tf.matmul(inputs2[i],Wxh)
                h_curr2 += b1
                h_curr2 = tf.sigmoid(h_curr2)

                h_curr1 = tf.sigmoid(tf.matmul(h_curr1,Whh) + tf.matmul(inputs1[i],Wxh) + b1)
                rnn_outputs1.append(h_curr1)
                rnn_outputs2.append(h_curr2)
        
        rnn_states = [tf.concat(1, [rnn_outputs1[i], rnn_outputs2[i]]) for i in range(self.config.max_steps)]
        return rnn_states

    def add_projection(self, rnn_states):
        # rnn_outputs is a list of length batch_size of lengths = seq_len. Where each list element is ??. I think.
        Whc = tf.get_variable('Whc', shape=(2*self.config.hidden_size,self.config.n_class))
        bhc = tf.get_variable('bhc', shape=(self.config.n_class,))
        projections = tf.matmul(rnn_states[-1],Whc) + bhc # in case we stop short sequences, the rnn_state in further time_steps should be unch
        return projections

    def add_loss_op(self, y):
        loss = tf.nn.softmax_cross_entropy_with_logits(y, self.labels)
        loss = tf.reduce_sum(loss)
        return loss
      
    def add_training_op(self, loss):
        #train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        train_op = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr).minimize(loss)
        return train_op

    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()

        print 'adsf __init__'
        print self.X1.get_shape
        self.inputs1, self.inputs2 = self.add_embedding()
        self.rnn_states            = self.add_model(self.inputs1, self.inputs2, self.seq_len1, self.seq_len2)
        self.projections           = self.add_projection(self.rnn_states)
        self.loss                  = self.add_loss_op(self.projections)
        self.train_step            = self.add_training_op(self.loss)
        self.predictions           = tf.argmax(tf.nn.softmax(self.projections),1)
        self.correct_predictions   = tf.equal(self.predictions,tf.argmax(self.labels,1))
        self.correct_predictions   = tf.reduce_sum(tf.cast(self.correct_predictions, 'int32'))
        

    def run_epoch(self, session, X1, X2, labels, train_op, verbose=10): # X and y are 2D np arrays
        print 'adsf run_epoch'
        config = self.config
        #state = tf.zeros([self.config.batch_size, self.config.hidden_size])
        state = self.initial_state.eval()
        data_len = np.shape(X1)[0]
        index = np.arange(data_len)
        np.random.shuffle(index)
        n_batches  = data_len // self.config.batch_size
        
        loss = 0.0
        for batch_num in range(n_batches):
            print 'sadf batch_num', str(batch_num)    
            x1_batch = X1[index[batch_num * self.config.batch_size : (batch_num+1) * self.config.batch_size], :]
            x2_batch = X2[index[batch_num * self.config.batch_size : (batch_num+1) * self.config.batch_size], :]
            seq_len_batch1 = [1 for i in range(X1.shape[0])]
            seq_len_batch2 = [1 for i in range(X1.shape[0])]
            labels_batch = labels[index[batch_num * self.config.batch_size : (batch_num+1) * self.config.batch_size]]
            print 'qwer', x1_batch.shape
            print 'qwer', x2_batch.shape
            feed_dict = {self.X1: x1_batch,
                         self.X2: x2_batch,
                         self.labels: labels_batch,
                         self.seq_len1: seq_len_batch1, 
                         self.seq_len2: seq_len_batch2} 
                         #self.initial_state: state}
            
            loss, total_correct, _ = session.run([self.loss, self.correct_predictions, train_op], feed_dict=feed_dict)
            total_loss.append(loss)
            
            if verbose and (batch_num+1)%verbose==0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(batch_num+1, n_batches, np.exp(np.mean(total_loss))))
                sys.stdout.flush()
            if verbose:
                sys.stdout.write('\r')
            
            return np.exp(np.mean(total_loss))

def run():
    config = Config()

    # We create the training model and generative model
    with tf.variable_scope('Model_RNN') as scope:
        model = Model_RNN(config)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
    print '1111'
    with tf.Session() as session:
        best_val_pp = float('inf')
        best_val_epoch = 0
        print '2222'
  
        session.run(init)
        print '3333'
        for epoch in xrange(config.max_epochs):
            print 'Epoch {}'.format(epoch)
            start = time.time()
            ###
            train_pp = model.run_epoch(
                session, model.encoded_train_1, model.encoded_train_2, model.labels_train,
                train_op=model.train_step)
            #valid_pp = model.run_epoch(session, model.encoded_valid)
            print 'Training perplexity: {}'.format(train_pp)
            #print 'Validation perplexity: {}'.format(valid_pp)
            #if valid_pp < best_val_pp:
                #best_val_pp = valid_pp
                #best_val_epoch = epoch
                #saver.save(session, './ptb_rnnlm.weights')
            #if epoch - best_val_epoch > config.early_stopping:
                #break
            print 'Total time: {}'.format(time.time() - start)

if __name__=='__main__':
    run()
