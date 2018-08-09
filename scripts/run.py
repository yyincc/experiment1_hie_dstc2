import data_utils as utils
from sklearn import metrics
from sklearn.model_selection import train_test_split
from models.hi_lstm import *
from dataset_walker import get_taskfile_db
from templatized import compare_with_golden, generate_RDL_data
from score import do_compute_score, do_load_json_result

from six.moves import range, reduce
import tensorflow as tf
import numpy as np
import time
import collections
import os
import json
from tqdm import tqdm
import pickle as pkl
import gensim
import gzip
import pickle
import random


tf.reset_default_graph()
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


del_all_flags(tf.flags.FLAGS)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Task
tf.flags.DEFINE_boolean("resetparam", True, "resetparam")


tf.flags.DEFINE_string("main_task", 'slot_tracking', "slot_tracking,action_chosen,both")
tf.flags.DEFINE_boolean("train", True, "training the model")
tf.flags.DEFINE_string("task", '2', "tasks 1-5")
tf.flags.DEFINE_string("testset", 'all', "testset 1-4 or all")
tf.flags.DEFINE_boolean("all_utter", False, "False for only use bot utterances ")
# Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 10, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("dp", 0.5, "dropout")
tf.flags.DEFINE_float("anneal_stop_epoch", 40, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 40, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
tf.flags.DEFINE_integer("encoder_hidden_units", 128, "Number of encoder_hidden_units")
tf.flags.DEFINE_integer("epochs", 30, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 300, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 100, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", 88, "Random state.")
# Methods Forced
tf.flags.DEFINE_boolean("speaker_info", False, "Add speaker information to embedding.")
tf.flags.DEFINE_boolean("user_in", False, "only use users' utterances.")
tf.flags.DEFINE_boolean("time_info_sents", False, "Add time information for per-response.")
tf.flags.DEFINE_boolean("word2vec", False, "whether to use pre-trained word2vec")
tf.flags.DEFINE_boolean("paragram", True, "whether to use pre-trained paragram")
tf.flags.DEFINE_boolean("usehalf", True, "use before index.")
tf.flags.DEFINE_boolean("combine", True, "combine.")
tf.flags.DEFINE_boolean("onlyuser", False, "onlyuser.")
# Methods
tf.flags.DEFINE_boolean("augment", False, "increase dataset based on origin one.")
tf.flags.DEFINE_boolean('rm_unk_sent', False, "Give unk sent lower ranking ")
# File Path
tf.flags.DEFINE_string("model_path", 'train-models/', "Directory containing database")
tf.flags.DEFINE_string("log_path", '../log/', "Directory log")
tf.flags.DEFINE_string("temp_path", '../data/processed/', "Directory of preprocessed data")
tf.flags.DEFINE_string("saved_name", '', "the name of saved model")
FLAGS = tf.flags.FLAGS

# preprocess the data

# please do not use the totality of the GPU memory
session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
session_config.gpu_options.per_process_gpu_memory_fraction = 0.90

taskchosen='2'
# create dircotory
if not os.path.exists(FLAGS.model_path): os.mkdir(FLAGS.model_path)
if not os.path.exists(FLAGS.model_path+'{}/'.format(taskchosen)): os.mkdir(FLAGS.model_path+'{}/'.format(taskchosen))
if not os.path.exists(FLAGS.log_path): os.mkdir(FLAGS.log_path)


train=json.load(open('../data/train.json','r'))
val=json.load(open('../data/dev.json','r'))
test=json.load(open('../data/test.json','r'))
testasr=json.load(open('../data/test_asr.json','r'))
cand=json.load(open('../data/cand.json','r'))

cand_idx=[]

cand_idx.append(dict((c, i + 2) for i, c in enumerate(cand['area'])))
cand_idx[0]['null']=0
cand_idx[0]['dontcare']=1
cand_idx.append(dict((c, i + 2) for i, c in enumerate(cand['food'])))
cand_idx[1]['null']=0
cand_idx[1]['dontcare']=1
cand_idx.append(dict((c, i + 2) for i, c in enumerate(cand['pricerange'])))
cand_idx[2]['null']=0
cand_idx[2]['dontcare']=1






if FLAGS.usehalf:
    for i in range(len(train)):
        train[i]['user']=train[i]['user'][0:train[i]['index']+1]
        train[i]['bot']=train[i]['bot'][0:train[i]['index']+1]
        train[i]['utterances']=train[i]['utterances'][0:2*(train[i]['index']+1)]
        train[i]['goal']=train[i]['goal'][0:train[i]['index']+1]
        train[i]['act']=train[i]['act'][0:train[i]['index']+1]
    for i in range(len(val)):
        val[i]['user']=val[i]['user'][0:val[i]['index']+1]
        val[i]['bot']=val[i]['bot'][0:val[i]['index']+1]
        val[i]['utterances']=val[i]['utterances'][0:2*(val[i]['index']+1)]
        val[i]['goal']=val[i]['goal'][0:val[i]['index']+1]
        val[i]['act']=val[i]['act'][0:val[i]['index']+1]
    for i in range(len(test)):
        test[i]['user']=test[i]['user'][0:test[i]['index']+1]
        test[i]['bot']=test[i]['bot'][0:test[i]['index']+1]
        test[i]['utterances']=test[i]['utterances'][0:2*(test[i]['index']+1)]
        test[i]['goal']=test[i]['goal'][0:test[i]['index']+1]
        test[i]['act']=test[i]['act'][0:test[i]['index']+1]
    for i in range(len(testasr)):
        testasr[i]['user']=testasr[i]['user'][0:testasr[i]['index']+1]
        testasr[i]['bot']=testasr[i]['bot'][0:testasr[i]['index']+1]
        testasr[i]['utterances']=testasr[i]['utterances'][0:2*(testasr[i]['index']+1)]
        testasr[i]['goal']=testasr[i]['goal'][0:testasr[i]['index']+1]
        testasr[i]['act']=testasr[i]['act'][0:testasr[i]['index']+1]


if FLAGS.onlyuser:
    for i in range(len(train)):
        train[i]['utterances']=train[i]['user']
        
    for i in range(len(val)):
        val[i]['utterances']=val[i]['user']
   
    for i in range(len(test)):
        test[i]['utterances']=test[i]['user']
    for i in range(len(testasr)):
        testasr[i]['utterances']=testasr[i]['user']
  

if FLAGS.combine:
    for i in range(len(train)):
        train[i]['utterances']=[]
        for j in range(len(train[i]['user'])):
            if train[i]['act'][j] == []:
                train[i]['utterances'].append([train[i]['user'][j][0]])
            else:
                train[i]['utterances'].append([train[i]['act'][j][0]+' '+train[i]['user'][j][0]])
        
    for i in range(len(val)):
        val[i]['utterances']=[]
        for j in range(len(val[i]['user'])):
            if val[i]['act'][j] == []:
                val[i]['utterances'].append([val[i]['user'][j][0]])
            else:
                val[i]['utterances'].append([val[i]['act'][j][0]+' '+val[i]['user'][j][0]])
    for i in range(len(test)):
        test[i]['utterances']=[]
        for j in range(len(test[i]['user'])):
            if test[i]['act'][j] == []:
                test[i]['utterances'].append([test[i]['user'][j][0]])
            else:
                test[i]['utterances'].append([test[i]['act'][j][0]+' '+test[i]['user'][j][0]])    
    for i in range(len(testasr)):
        testasr[i]['utterances']=[]
        for j in range(len(testasr[i]['user'])):
            if testasr[i]['act'][j] == []:
                testasr[i]['utterances'].append([testasr[i]['user'][j][0]])
            else:
                testasr[i]['utterances'].append([testasr[i]['act'][j][0]+' '+testasr[i]['user'][j][0]])    


    

utils.process(train)
utils.process(val)
utils.process(test)
utils.process(testasr)

#if FLAGS.speaker_info:
#    for i in range(len(train)):
#        for j in range(len(train[i]['user'])):
#            train[i]['user'][j]= ["$user"] + train[i]['user'][j]
#        for j in range(len(train[i]['bot'])):
#            train[i]['bot'][j]= ["$bot"] + train[i]['bot'][j]
#            
#    for i in range(len(val)):
#        for j in range(len(val[i]['user'])):
#            val[i]['user'][j]= ["$user"] + val[i]['user'][j]
#        for j in range(len(val[i]['bot'])):
#            val[i]['bot'][j]= ["$bot"] + val[i]['bot'][j]
#            
#    for i in range(len(test)):
#        for j in range(len(test[i]['user'])):
#            test[i]['user'][j]= ["$user"] + test[i]['user'][j]
#        for j in range(len(test[i]['bot'])):
#            test[i]['bot'][j]= ["$bot"] + test[i]['bot'][j]
#            
#    for i in range(len(testasr)):
#        for j in range(len(testasr[i]['user'])):
#            testasr[i]['user'][j]= ["$user"] + testasr[i]['user'][j]
#        for j in range(len(testasr[i]['bot'])):
#            testasr[i]['bot'][j]= ["$bot"] + testasr[i]['bot'][j]



#if FLAGS.combine:
#    for i in range(len(train)):
#        train[i]['utterances']=[]
#        for j in range(len(train[i]['user'])):
#            train[i]['utterances'].append(train[i]['bot'][j]+train[i]['user'][j])
#        
#    for i in range(len(val)):
#        val[i]['utterances']=[]
#        for j in range(len(val[i]['user'])):
#            val[i]['utterances'].append(val[i]['bot'][j]+val[i]['user'][j])
#    for i in range(len(test)):
#        test[i]['utterances']=[]
#        for j in range(len(test[i]['user'])):
#            test[i]['utterances'].append(test[i]['bot'][j]+test[i]['user'][j])    
#    for i in range(len(testasr)):
#        testasr[i]['utterances']=[]
#        for j in range(len(testasr[i]['user'])):
#            testasr[i]['utterances'].append(testasr[i]['bot'][j]+testasr[i]['user'][j])  
            
#if FLAGS.combine:
#    for i in range(len(train)):
#        train[i]['utterances']=[]
#        for j in range(len(train[i]['user'])):
#            train[i]['utterances'].append([train[i]['bot'][j][0]+' '+train[i]['user'][j][0]])
#        
#    for i in range(len(val)):
#        val[i]['utterances']=[]
#        for j in range(len(val[i]['user'])):
#            val[i]['utterances'].append([val[i]['bot'][j][0]+' '+val[i]['user'][j][0]])
#    for i in range(len(test)):
#        test[i]['utterances']=[]
#        for j in range(len(test[i]['user'])):
#            test[i]['utterances'].append([test[i]['bot'][j][0]+' '+test[i]['user'][j][0]])    
#    for i in range(len(testasr)):
#        testasr[i]['utterances']=[]
#        for j in range(len(testasr[i]['user'])):
#            testasr[i]['utterances'].append([testasr[i]['bot'][j][0]+' '+testasr[i]['user'][j][0]])    

if FLAGS.resetparam:
    vocab, word_idx, max_story_size, mean_story_size, sentence_size = utils.data_information([train,val,test,testasr],FLAGS)
    
    vocab_size = len(word_idx) + 1 # +1 for nil word (0 for nil)
    sentence_size = sentence_size + 5 # add some space for testing data
    memory_size = min(FLAGS.memory_size, max_story_size)
    loaded_embeddings=None
    
    word_idx=pkl.load(open('word_idx.pkl','rb'))
    loaded_embeddings=pkl.load(open('loaded_embeddings.pkl','rb'))
    
    
#    if FLAGS.word2vec:
#        loaded_embeddings = utils.loadEmbedding_rand('/Users/yangyang/Dialog project/yy-dstc6/scripts/GoogleNews-vectors-negative300.bin', word_idx,True)
#    if FLAGS.paragram:
#        loaded_embeddings = utils.loadEmbedding_rand('/Users/yangyang/Dialog project/yy-dstc6/scripts/paragram999.txt', word_idx,False)
#    else:
#        loaded_embeddings=utils.loadEmbedding_rand(None, word_idx,True)        

    # vectorize data
    trainS, trainA = utils.vectorize_data(train, word_idx, sentence_size, FLAGS.batch_size, memory_size, cand_idx,FLAGS)
    valS,  valA= utils.vectorize_data(val, word_idx, sentence_size, FLAGS.batch_size, memory_size, cand_idx,FLAGS)
    testS,  testA= utils.vectorize_data(test, word_idx, sentence_size, FLAGS.batch_size, memory_size, cand_idx,FLAGS)
    testasrS,  testasrA= utils.vectorize_data(testasr, word_idx, sentence_size, FLAGS.batch_size, memory_size, cand_idx,FLAGS)
    
    
    f_param = open(FLAGS.model_path+'param','wb')
    pkl.dump([testasrS,  testasrA,testasr,memory_size,train,val,test,word_idx,sentence_size,memory_size, vocab_size,trainS,valS,trainA,valA,loaded_embeddings,testS, testA], f_param)
    f_param.close()

else:
    if os.path.exists(FLAGS.model_path+'param'):
        f_param = open(FLAGS.model_path+'param','rb')
        testasrS,  testasrA,testasr,memory_size,train,val,test,word_idx,sentence_size,memory_size, vocab_size,trainS,valS,trainA,valA,loaded_embeddings,testS, testA= pkl.load(f_param)
    else:
        print ('[ERROR] No param is stored...')
        exit(1)    

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size
total_batch = int(len(trainS) / batch_size)
jj=int(len(trainS)!=total_batch*batch_size)   

batches=[(batch_size * i,batch_size * (i + 1)) if batch_size * (i + 1) <= len(trainS) else (batch_size * i,len(trainS)) for i in range(total_batch+jj)]

n_val =len(valS)
n_test =len(testS)
n_testasr =len(testasrS)

with tf.Session(config=session_config) as sess:
    model = hierarchical_lstm(vocab_size=vocab_size, sent_len=sentence_size, sent_numb=memory_size, 
                              encoder_hidden_units=FLAGS.encoder_hidden_units, embedding_size=FLAGS.embedding_size,
                              L2=0, clip_gradients=FLAGS.max_grad_norm,session=sess,embeddings=loaded_embeddings)

    start_time = time.time()
    if FLAGS.train:
        
        print('start')

        saver = tf.train.Saver()
        best_acc_val = 0
        cnt = 0
        cnt_one = 0
        for t in range(1, FLAGS.epochs+1):
            # Stepped learning rate
            if t - 1 <= FLAGS.anneal_stop_epoch:
                anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
            else:
                anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
            lr = FLAGS.learning_rate / anneal
            # data shuffling
            dialog=list(zip(trainS,trainA))
            random.shuffle(dialog)
            trainS,trainA=zip(*dialog)

            train_labels, val_labels,test_labels,testasr_labels = trainA, np.array(valA),np.array(testA),np.array(testasrA)
            total_cost,total_acc,index_bat = 0.0, 0.0, 0
         #   total_cost_cui=0
            prog_bar = tqdm(batches)
            for start, end in prog_bar:
                index_bat += 1
                s = trainS[start:end]
                a = trainA[start:end]
                cost_t,acc_t = model.batch_fit(s,  a, lr,FLAGS.dp)
                total_cost += cost_t
            #    total_cost_cui += cost_t_cui
                total_acc += acc_t
                prog_bar.set_description('Acc: {:10.4f} Loss: {:10.4f}'.format(total_acc/index_bat,total_cost/index_bat))

            if t % FLAGS.evaluation_interval == 0:
                val_preds, val_preds_prob, val_loss,pred_cui,pred_loc,pred_pri = utils.batch_evaluate(model, valS, valA, n_val, batch_size) 

                val_acc = metrics.accuracy_score(val_preds==val_labels,np.ones((n_val,3)))
                
                val_acc_cui = metrics.accuracy_score(pred_cui==val_labels[:,1][:,None],np.ones((n_val,1)))
                val_acc_loc = metrics.accuracy_score(pred_loc==val_labels[:,0][:,None],np.ones((n_val,1)))
                val_acc_pri = metrics.accuracy_score(pred_pri==val_labels[:,2][:,None],np.ones((n_val,1)))

                
                print ('---------val--------------')
                print ('Epoch', t, 'joint Loss:', val_loss, 'joint Acc:', val_acc)
                print ('Epoch', t, 'cui Acc:', val_acc_cui, 'loc Acc:', val_acc_loc,'pri Acc:', val_acc_pri)
                
                test_preds, test_preds_prob, test_loss,test_cui,test_loc,test_pri = utils.batch_evaluate(model, testS, testA, n_test, batch_size) 

                test_acc = metrics.accuracy_score(test_preds==test_labels,np.ones((n_test,3)))
                
                test_acc_cui = metrics.accuracy_score(test_cui==test_labels[:,1][:,None],np.ones((n_test,1)))
                test_acc_loc = metrics.accuracy_score(test_loc==test_labels[:,0][:,None],np.ones((n_test,1)))
                test_acc_pri = metrics.accuracy_score(test_pri==test_labels[:,2][:,None],np.ones((n_test,1)))

                
                print ('---------test--------------')
                print ('Epoch', t, 'joint Loss:', test_loss, 'joint Acc:', test_acc)
                print ('Epoch', t, 'cui Acc:', test_acc_cui, 'loc Acc:', test_acc_loc,'pri Acc:', test_acc_pri)
                
                
                testasr_preds, testasr_preds_prob, testasr_loss,testasr_cui,testasr_loc,testasr_pri = utils.batch_evaluate(model, testasrS, testasrA, n_testasr, batch_size) 

                testasr_acc = metrics.accuracy_score(testasr_preds==testasr_labels,np.ones((n_testasr,3)))
                
                testasr_acc_cui = metrics.accuracy_score(testasr_cui==testasr_labels[:,1][:,None],np.ones((n_testasr,1)))
                testasr_acc_loc = metrics.accuracy_score(testasr_loc==testasr_labels[:,0][:,None],np.ones((n_testasr,1)))
                testasr_acc_pri = metrics.accuracy_score(testasr_pri==testasr_labels[:,2][:,None],np.ones((n_testasr,1)))

                
                print ('---------testasr--------------')
                print ('Epoch', t, 'joint Loss:', testasr_loss, 'joint Acc:', testasr_acc)
                print ('Epoch', t, 'cui Acc:', testasr_acc_cui, 'loc Acc:', testasr_acc_loc,'pri Acc:', testasr_acc_pri)
                
                


            if(val_acc >= best_acc_val):
                best_acc_val = val_acc
                cnt = 0
                model.saver.save(model._sess, FLAGS.model_path + '{}/'+FLAGS.saved_name+'hi_lstm_model.ckpt'.format(taskchosen), 
                    global_step=t)
                print ('[Saving the model at val acc %s...]'%(str(val_acc)))
            else:
                cnt += 1
            print ('COUNT VALUE %d' % int(cnt))
            print ('-----------------------')
            if val_acc == 1.0:
                cnt_one += 1
            if val_acc == 1.0 and cnt_one == 3:
                break
            if cnt>= 10:
                break
    
    # restore checkpoint
    ckpt = tf.train.latest_checkpoint(FLAGS.model_path+str(taskchosen))
    if ckpt:
        print ('>> restoring checkpoint from', ckpt)
        model.saver.restore(model._sess, ckpt)

