from __future__ import print_function
import numpy as np
import tensorflow as tf
from tflearn.activations import sigmoid, softmax
from tensorflow.python.framework import ops
from functools import partial
import logging
import datetime


def attention_softmax2d(values):
    """
    Performs a softmax over the attention values.

    :param values: 3d tensor with raw values
    :return: 3d tensor, same shape as input
    """
  
    softmaxes = tf.nn.softmax(values)
    return softmaxes

def mask_2d(values, sentence_sizes, mask_value):
    """
    Given a batch of matrices, each with shape m x n, mask the values in each
    row after the positions indicated in sentence_sizes.

    This function is supposed to mask the last columns in the raw attention
    matrix (e_{i, j}) in cases where the sentence2 is smaller than the
    maximum.

    :param values: tensor with shape (batch_size, m, n)
    :param sentence_sizes: tensor with shape (batch_size) containing the
        sentence sizes that should be limited
    :param mask_value: scalar value to assign to items after sentence size
    :param dimension: over which dimension to mask values
    :return: a tensor with the same shape as `values`
    """
  
    time_steps1 = tf.shape(values)[1]

    ones = tf.ones_like(values, dtype=tf.float32)
    pad_values = mask_value * ones
    mask = tf.sequence_mask(sentence_sizes, time_steps1)

    # mask is (batch_size, sentence2_size). we have to tile it for 3d
  

    masked = tf.where(mask, values, pad_values)

  
    return masked

class hierarchical_lstm():
    def __init__(self, vocab_size, sent_len, sent_numb, encoder_hidden_units, embedding_size,
                 L2,clip_gradients,session=tf.Session(),num_atm=3,num_cui=93,num_loc=7,num_peo=4,num_pri=5,
                initializer=tf.random_normal_initializer(stddev=0.1),embeddings=None):
        """
        Initialize an Entity Network with the necessary hyperparameters.

        :param vocabulary: Word Vocabulary for given model.
        :param sentence_len: Maximum length of a sentence.
        :param story_len: Maximum length of a story.
        """
        self.vocab_size, self.sent_len, self.sent_numb = vocab_size, sent_len, sent_numb
        self.embedding_size, self.encoder_hidden_units, self.init = embedding_size, encoder_hidden_units, initializer
        self.num_atm,self.num_cui,self.num_loc,self.num_peo,self.num_pri = num_atm,num_cui,num_loc,num_peo,num_pri
        self.opt = 'Adam'
        self.clip_gradients = clip_gradients
        self.L2 = L2
        ## setup placeholder
        self.S = tf.placeholder(tf.int32, shape=[None,None,self.sent_len],name="Story")
        self.Q = tf.placeholder(tf.int32, shape=[None,self.sent_len],name="Question")
        self.A = tf.placeholder(tf.int32, shape=[None,3],name="Answer")
        self.dropout=tf.placeholder(tf.float32, [], 'dropout')
        self.visualization = False
        
        self.embeddings=embeddings
        
        # self.keep_prob = tf.placeholder(tf.float32, name= "dropout")
        self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
        self.batch_size = tf.shape(self.S)[0]

        # Setup Global, Epoch Step
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        # Instantiate Network Weights
        self.E =  tf.Variable(embeddings, trainable=False,name='emb')

        # Build Inference Pipeline
        self.logits_cui,self.logits_loc,self.logits_pri = self.inference()

        # Build Loss Computation
        self.loss_op = self.loss()

        # Build Training Operation
        self.train_op = self.train()

        self.saver = tf.train.Saver(max_to_keep=5)

        # Create operations for computing the accuracy
        self.correct_prediction_cui = tf.equal(tf.argmax(self.logits_cui, 1,output_type=tf.int32),self.A[:,1])
        self.correct_prediction_loc = tf.equal(tf.argmax(self.logits_loc, 1,output_type=tf.int32),self.A[:,0])
        self.correct_prediction_pri = tf.equal(tf.argmax(self.logits_pri, 1,output_type=tf.int32),self.A[:,2])

        self.correct_prediction = tf.reduce_all([self.correct_prediction_loc,self.correct_prediction_cui,self.correct_prediction_pri])
        #tf.equal([tf.argmax(self.logits_cui, 1,output_type=tf.int32),tf.argmax(self.logits_loc, 1,output_type=tf.int32),tf.argmax(self.logits_peo, 1,output_type=tf.int32),tf.argmax(self.logits_pri, 1,output_type=tf.int32)],self.A)

        self.accuracy_cui = tf.reduce_mean(tf.cast(self.correct_prediction_cui, tf.float32), name="Accuracy_cui")
        self.accuracy_loc = tf.reduce_mean(tf.cast(self.correct_prediction_loc, tf.float32), name="Accuracy_loc")
        self.accuracy_pri = tf.reduce_mean(tf.cast(self.correct_prediction_pri, tf.float32), name="Accuracy_pri")

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="Accuracy")

        
        # predict op 
        predict_op_cui = tf.argmax(self.logits_cui, 1, name="predict_op_cui")
        predict_op_loc = tf.argmax(self.logits_loc, 1, name="predict_op_loc")
        predict_op_pri = tf.argmax(self.logits_pri, 1, name="predict_op_pri")
        
        predict_proba_op_cui = tf.nn.softmax(self.logits_cui, name="predict_proba_op_cui")
        predict_proba_op_loc = tf.nn.softmax(self.logits_loc, name="predict_proba_op_loc")
        predict_proba_op_pri = tf.nn.softmax(self.logits_pri, name="predict_proba_op_pri")
        
        predict_log_proba_op_cui = tf.log(predict_proba_op_cui, name="predict_log_proba_op_cui")
        predict_log_proba_op_loc = tf.log(predict_proba_op_loc, name="predict_log_proba_op_loc")
        predict_log_proba_op_pri = tf.log(predict_proba_op_pri, name="predict_log_proba_op_pri")
        

        self.predict_op_cui = predict_op_cui
        self.predict_op_loc = predict_op_loc
        self.predict_op_pri = predict_op_pri
        self.predict_op=tf.stack([predict_op_loc,predict_op_cui,predict_op_pri],1)
        
        
        self.predict_proba_op_cui = tf.pad(predict_proba_op_cui,tf.constant([[0, 0,], [0, 93-self.num_cui]]))
        self.predict_proba_op_loc = tf.pad(predict_proba_op_loc,tf.constant([[0, 0,], [0, 93-self.num_loc]]))
        self.predict_proba_op_pri = tf.pad(predict_proba_op_pri,tf.constant([[0, 0,], [0, 93-self.num_pri]]))
        self.predict_proba_op=tf.stack([self.predict_proba_op_cui,self.predict_proba_op_loc,self.predict_proba_op_pri],1)
        
        self.predict_log_proba_op_cui = tf.pad(predict_log_proba_op_cui,tf.constant([[0, 0,], [0, 93-self.num_cui]]))
        self.predict_log_proba_op_loc = tf.pad(predict_log_proba_op_loc,tf.constant([[0, 0,], [0, 93-self.num_loc]]))
        self.predict_log_proba_op_pri = tf.pad(predict_log_proba_op_pri,tf.constant([[0, 0,], [0, 93-self.num_pri]]))
        self.predict_log_proba_op=tf.stack([self.predict_log_proba_op_cui,self.predict_log_proba_op_loc,self.predict_log_proba_op_pri],1)

        parameters = self.count_parameters()
        logging.info('Parameters: {}'.format(parameters))
        
        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)
        
        self.cnt = 0




    def inference(self):
        """
        Build inference pipeline, going from the story and question, through the memory cells, to the
        distribution over possible answers.
        """

        # Story Input Encoder
        story_embeddings = tf.nn.embedding_lookup(self.E, self.S) # Shape: [None, story_len, sent_len, embed_sz]
        words_embeddings=tf.reshape(story_embeddings,[-1,self.sent_len,self.embedding_size])       #Shape: [None x story_len, sent_len, embed_sz]
        words_embeddings = tf.nn.dropout(words_embeddings,self.dropout)
        words_cell_bw=tf.contrib.rnn.BasicLSTMCell(self.encoder_hidden_units)
        words_cell_fw=tf.contrib.rnn.BasicLSTMCell(self.encoder_hidden_units)
        self.sent_length=self.get_sentence_length()
        bi_word_outputs, bi_word_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=words_cell_fw,
                                    cell_bw=words_cell_bw,
                                    inputs=words_embeddings,
                                    sequence_length=self.sent_length,
                                    dtype = tf.float32,
                                    swap_memory=True)          #Shape: [None x story_len, num_units]
        bi_word_outputs=tf.concat(bi_word_outputs, -1)
        
#        score_=tf.layers.dense(bi_word_outputs,activation=tf.tanh, units=100,name='www')
#        score=tf.layers.dense(score_, units=1,activation=None,use_bias=False,name='vvv')
#        self.masked=mask_2d(score[:,:,0],self.sent_length,-np.inf)
#        self.att= mask_2d(attention_softmax2d(self.masked),self.sent_length,0)
#        sent_inputs=tf.matmul(tf.transpose(bi_word_outputs,[0,2,1]),tf.expand_dims(self.att,2))[:,:,0]
#        sent_inputs=tf.reshape(sent_inputs,([self.batch_size,self.sent_numb,self.encoder_hidden_units*2]))
#        
#        
        
        bi_word_state=(tf.concat([bi_word_state[0][0],bi_word_state[1][0]],-1),tf.concat([bi_word_state[0][1],bi_word_state[1][1]],-1))
        sent_inputs=tf.contrib.rnn.LSTMStateTuple(tf.reshape(bi_word_state[0],[self.batch_size,self.sent_numb,self.encoder_hidden_units*2]),
                                                   tf.reshape(bi_word_state[1],[self.batch_size,self.sent_numb,self.encoder_hidden_units*2]))     #Shape: [None , story_len, 2*num_units]


        sinputs=tf.nn.dropout(sent_inputs[1],self.dropout)
        self.story_length = self.get_story_length()
        sent_cell=tf.contrib.rnn.BasicLSTMCell(self.encoder_hidden_units*2)
        sent_outputs, sent_state=tf.nn.dynamic_rnn(sent_cell,
                                    inputs=sinputs,
                                    sequence_length=self.story_length,
                                    dtype = tf.float32,
                                    swap_memory=True)

        #logits_atm=tf.layers.dense(sent_state, units=self.num_atm)
        
        sent_s=tf.nn.dropout(sent_state[1],self.dropout)
        
        logits_cuii=tf.layers.dense(sent_s, units=100,name='cc')
        logits_locc=tf.layers.dense(sent_s, units=100,name='ll')
        logits_prii=tf.layers.dense(sent_s, units=100,name='prr')

        logits_cui=tf.layers.dense(logits_cuii, units=self.num_cui,name='c')
        logits_loc=tf.layers.dense(logits_locc, units=self.num_loc,name='l')
        logits_pri=tf.layers.dense(logits_prii, units=self.num_pri,name='pr')

    

        return logits_cui,logits_loc,logits_pri
    

    def loss(self):
        """
        Build loss computation - softmax cross-entropy between logits, and correct answer.
        """
        #cross_entropy_atm = tf.tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_atm, labels=self.A[], name="cross_entropy_atm")
        cross_entropy_cui = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_cui, labels=self.A[:,1], name="cross_entropy_cui")
        cross_entropy_loc = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_loc, labels=self.A[:,0], name="cross_entropy_loc")
        cross_entropy_pri = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_pri, labels=self.A[:,2], name="cross_entropy_pri")
        cross_entropy_sum = tf.reduce_mean([cross_entropy_cui,cross_entropy_loc,cross_entropy_pri], name="cross_entropy_sum")
        print(cross_entropy_cui)
       # self.loss_cui,self.loss_loc,self.loss_peo,self.loss_pri = tf.reduce_sum(cross_entropy_cui),tf.reduce_sum(cross_entropy_loc),tf.reduce_sum(cross_entropy_peo),tf.reduce_sum(cross_entropy_pri)
        if(self.L2 !=0.0):
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'rnn/DynamicMemoryCell/biasU:0' != v.name ])  * self.L2
            # lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in var])  * self.L2
            return cross_entropy_sum + lossL2
            # return tf.losses.sparse_softmax_cross_entropy(self.A,self.logits)+lossL2
        else:
            return cross_entropy_sum
            # return tf.losses.sparse_softmax_cross_entropy(self.A,self.logits)

    def train(self):
        """
        Build Optimizer Training Operation.
        """
        # learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
        #                                            self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_op, global_step=self.global_step,
                                                   learning_rate=self.learning_rate, optimizer=self.opt,
                                                   clip_gradients=self.clip_gradients)
        return train_op

    def get_story_length(self):
        """
        This is a hacky way of determining the actual length of a sequence that has been padded with zeros.
        """
        used = tf.sign(tf.reduce_max(tf.abs(self.S), axis=-1))
        story_length = tf.cast(tf.reduce_sum(used, axis=-1), tf.int32)
        return story_length

    def get_sentence_length(self):
        SS=tf.reshape(self.S,[-1,self.sent_len]) 
        used = tf.sign(tf.abs(SS))
        sent_length = tf.cast(tf.reduce_sum(used, axis=-1), tf.int32)
        return sent_length


    def position_encoding(self,sentence_size, embedding_size):
        """
        Position Encoding described in section 4.1 [1]
        """
        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = sentence_size+1
        le = embedding_size+1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        return np.transpose(encoding)

    def count_parameters(self):
        "Count the number of parameters listed under TRAINABLE_VARIABLES."
        num_parameters = sum([np.prod(tvar.get_shape().as_list())
                              for tvar in tf.trainable_variables()])
        return num_parameters
    

    def batch_fit(self, stories, answers, learning_rate, cand=[],dp=0.7):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)
        Returns:
            loss: floating-point number, the loss computed for the batch
        """

        feed_dict = {self.S: stories, self.A: answers, self.learning_rate: learning_rate,self.dropout:dp}
        loss, _, acc = self._sess.run([self.loss_op, self.train_op, self.accuracy], feed_dict=feed_dict)
        return loss, acc
#self.loss_cui,self.loss_loc,self.loss_peo,self.loss_pri,
        
    def predict(self, stories, answers=[], cand=[], word_in = [],dp=1):
        feed_dict = {self.S: stories,self.dropout:dp}
        if self.visualization:
            self.viz(stories,queries,feed_dict,word_in)
        if answers!=[]:
            feed_dict[self.A] = answers
            return self._sess.run([self.predict_op, self.loss_op,self.predict_op_cui,self.predict_op_loc,self.predict_op_pri], feed_dict=feed_dict)
        else:
            return self._sess.run(self.predict_op, feed_dict=feed_dict)

    def predict_proba(self, stories, cand=[],dp=1):
        feed_dict = {self.S: stories,self.dropout:dp}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, stories,dp=1):
        feed_dict = {self.S: stories}
        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)


    def sigmoid(self,x):
        x = np.array(x, dtype=np.float128)
        return 1 / (1 + np.exp(-x))

    def viz(self,mb_x1,mb_x2,dic,word_ind):
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='Times-Roman')
        sns.set_style(style='white')
        self.cnt += 1

        idx2candid = dict((i, c) for c, i in word_ind.items())
        s_s=[]
        ### mb_x1 input senteces
        for m in mb_x1[0]:
            t = []
            for e in m:
                if(e != 0):
                    if (idx2candid[e][0]=='#'):
                        temp = idx2candid[e].split('_')
                        if(temp[1][:3]=='res'):
                            t.append("[NAME"+temp[2][0]+"]")
                        else:
                            t.append("["+temp[1][:3].upper()+temp[2][0]+"]")
                    elif idx2candid[e][0]=='<':
                        t.append(idx2candid[e][1:-1])
                    elif idx2candid[e][:2]=='r_':
                        t.append(idx2candid[e][2:])
                    elif idx2candid[e][:3]=='api':
                        t.append(idx2candid[e][:3]+r'\_'+ idx2candid[e][4:])
                    else:
                        t.append(idx2candid[e])
            if(len(t) > 0):
                s_s.append(t[2:])

        s_s = [" ".join(sss) for sss in s_s]

        q_q = []
        for e2 in mb_x2[0]:
            if(e2 != 0):
                if (idx2candid[e2][0]=='#'):
                    temp = idx2candid[e2].split('_')
                    q_q.append("["+temp[1][:3]+'-'+temp[2][0]+"]")
                elif idx2candid[e2][0]=='<':
                    q_q.append(idx2candid[e2][1:-1])
                elif idx2candid[e2][:2]=='r_':
                    q_q.append(idx2candid[e2][2:])
                else:
                    q_q.append(idx2candid[e2])


        q_q = " ".join(q_q[2:])

        k,o,s,q,l,E = self._sess.run([self.keys,
                                    self.out,
                                    self.story_embeddings,
                                    self.query_embedding,
                                    self.length,
                                    self.E],feed_dict=dic)

        gs=[]
        for i in range(int(l[0])):
            temp = np.split(o[0][i], len(k))
            g =[]
            for j in range(len(k)):
                # a = np.argmax(np.matmul(E,k[j]))
                # print(idx2candid[a])
                # print(np.argmax(np.matmul(E,k[j])))
                # print(np.max(np.matmul(E,k[j])))             
                # g.append(sigmoid(np.inner(s[0][i],temp[j])+np.inner(s[0][i],k[j])+np.inner(s[0][i],q[0][0])))
                g.append(self.sigmoid(np.inner(s[0][i],temp[j])+np.inner(s[0][i],k[j])))
            gs.append(g)

        plt.figure(figsize=(5,7.5))
        ax = sns.heatmap(np.array(gs),cmap="YlGnBu",vmin=0, vmax=1,cbar=False)
        
        ax.set_yticks([i+0.5 for i in range(len(s_s))],minor=True)
        ax.set_yticklabels(s_s,rotation=0,fontsize=7)
        ax.set_xticklabels([ i+1 for i in range(len(k)) ],rotation=0 )

        plt.title(q_q,fontsize=7)
        plt.tight_layout()
        plt.subplots_adjust(left=0.75, right=0.99, top=0.96, bottom=0.4)

        plt.savefig('../data/plot/%s.pdf'%str(self.cnt), format='pdf', dpi=300)
        plt.close()

def prelu(features, alpha, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU'):
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg

def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))



