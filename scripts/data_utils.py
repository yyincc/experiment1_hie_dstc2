
import numpy as np
import json
from itertools import chain
import re
from sklearn.model_selection import train_test_split
import operator
from copy import deepcopy
from functools import reduce
from spacy.lang.en import English
from score import do_compute_score, do_load_json_result
import gensim
from tqdm import tqdm
### The current dialog format
### [{dialog_id : " ",
#     utterances: [" "],
#     candidates: [{candidate_id: " " , utterance: ""}, ... ],
#     answer: {candidate_id: " " , utterance: ""} ]
nlp=English()
def load_task(path_to_data, FLAGS, cand_idx_out={}):
    json_data = []
    nlp=English()
    for task_dir in path_to_data:
        fd = open(task_dir, 'rb')
        json_data += json.load(fd)
        fd.close()
    train = json_data
    
    train = get_stories(train, FLAGS.speaker_info, FLAGS.time_info_sents,user_in=FLAGS.user_in)
    
    if FLAGS.main_task=='slot_tracking':
        train=[t for t in train if str(t['a'][0]) == 'api_call']

#    train, val = train_test_split(train, test_size=0, random_state=FLAGS.random_state)  


    return train, []

def process(dataset):

    for id in range(len(dataset)):
        utterances=dataset[id]['utterances']
        user=dataset[id]['user']
        bot=dataset[id]['bot']
        
        dataset[id]['utterances']=[]
        dataset[id]['user']=[]
        dataset[id]['bot']=[]
        for i in range(len(utterances)):
            dataset[id]['utterances'].append(tokenize(utterances[i][0]))
        
        for i in range(len(user)):
            dataset[id]['user'].append(tokenize(user[i][0]))
        
        for i in range(len(bot)):
            dataset[id]['bot'].append(tokenize(bot[i][0]))
            


            
            
def load_test(path_to_data,FLAGS):

    json_data = []
    for task_dir in path_to_data:
        fd = open(task_dir, 'rb')
        json_data += json.load(fd)
        fd.close()
    test = get_stories(json_data, FLAGS.speaker_info, FLAGS.time_info_sents,user_in=FLAGS.user_in)
    
    if FLAGS.main_task=='slot_tracking':
        test=[t for t in test if str(t['a'][0]) == 'api_call']


    return test

def loadEmbedding_rand(path, word_indices, binary,divident = 1.0): # TODO double embedding
    """
    Load embeddings. Doing a random normal initialization for OOV words.
    """

    n = len(word_indices) 
    m = 300
    emb = np.empty((n+1, m), dtype=np.float32)
    emb[:,0:300] = np.random.normal(size=(n+1,300)) / divident
    emb[0, :] = np.zeros((1,m), dtype="float32") 
    
    if path !=None:
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)
    
        for word in word_indices.keys():
            try:
                emb[word_indices[word],  0:300] = model[word]
            except KeyError:
                print(word)
                continue
    
                

    return emb



def get_stories(json_data,speaker_info, time_info_sents, user_in=False, testing=False):
    '''Parse stories provided in the tasks format
    '''
    data = []
    for story in tqdm(json_data):
        utterances_origin = story['utterances']
        utterances_list = []
        a_list = []

        utterances_list.append(utterances_origin)
        a_list.append( tokenize(story['answer']['utterance']) )
       
            
        utterances_list_counter = 0
        for utterances in utterances_list:
            storyInfo = {}
            utter_list = []
            history=[]
            utter_counter = 0
            speaker_type_list, time_list = get_add_info(utterances)
            for utter in utterances[:len(utterances)-1]:
                add_item = []
                if time_info_sents: add_item += [time_list[utter_counter]]
                if speaker_info: add_item += [speaker_type_list[utter_counter]]
                utter_list.append(add_item+[token.text for token in nlp(utter)])
                history.append(tokenize(utter))
                if user_in:
                    if '$user' in add_item:
                        utter_list.append(tokenize(utter))
                else:    
                    utter_list.append(add_item+tokenize(utter))
                utter_counter += 1
            
            # add item to last question from user
            add_item_q = []
            if time_info_sents: add_item_q += [time_list[-1]]
            if speaker_info: add_item_q += ['speaker_user']
            #q = add_item_q + [token.text for token in nlp(utterances[len(utterances)-1])]
            q = add_item_q + tokenize(utterances[len(utterances)-1])

            a = a_list[utterances_list_counter] if not testing else [] 

            

            
            storyInfo['dialog_id'] = story['dialog_id']
            storyInfo['history'] = history
            storyInfo['utterances'] = utter_list
            storyInfo['a'] = a
            data.append(storyInfo)
            utterances_list_counter += 1
    return data


def get_add_info(utterances): 
    """
    return who speaks the utterance and when is it
    """
    time_list = []
    speaker_type_list = []
    speaker_type = ["$user", "$bot", '$options']
    start = 0
    counter = 0
    for utter in utterances:
        token = tokenize(utter)
        if is_option(token):
            speaker_type_list.append(speaker_type[2])
            time_list.append('time-api')
        else:
            time_list.append('time{}'.format( int(np.ceil((counter+1)*1.0/2)) ) )
            speaker_type_list.append(speaker_type[counter%2])
            counter += 1
    return speaker_type_list, time_list

def get_cand2idx(data, all_utter=False):
    """
    return dictionary of candidates bot answers
    """
    cand_list = []
    for d in data:
        c = [ x['utterance'] for x in d['cand_list'] ]
        a = d['a'][:-1]
        utters = d['utter_list']
        for c_i in c:
            if 'unk' not in c_i:
                cand_list.append(" ".join(c_i))
        cand_list.append(" ".join(a))
        for utter in utters:
            if '$user' in utter or '$options' in utter:
                if all_utter and '$user' in utter:
                    cand_list.append("$ "+" ".join(utter[2:]))
                continue
            else:
                cand_list.append(" ".join(utter[2:]))
    cand_list = list(set(cand_list))
    cand_idx = dict((c, i+1) for i, c in enumerate(cand_list))
    cand_idx['UNK-SENT'] = 0
    candidates = []
    for cand in cand_idx.keys():
        candidates.append(tokenize(cand))
    return cand_idx

def is_option(token_sentence):
    if len(token_sentence) == 3 and ('r_' in token_sentence[1][:2] or 'R_' in token_sentence[1][:2] ):
        return True
    else:
        return False

def get_story_cand_idx(data, cand_idx):
    cand_per_story = []
    n_cand = len(cand_idx)
    for d in data:
        c = d['cand_list']
        temp = []
        for c_i in c:
            if 'unk' not in c_i['utterance']:
                str_utter =" ".join(c_i['utterance'])
                if str_utter not in cand_idx:
                    max_score_idx = find_similar_candidate(cand_idx, c_i['utterance'])
                    temp.append( { 'candidate_id': c_i['candidate_id'], 'utterance': c_i['utterance'] ,'idx': max_score_idx } )
                else:
                    temp.append( { 'candidate_id': c_i['candidate_id'], 'utterance': c_i['utterance'] ,'idx': cand_idx[" ".join(c_i['utterance'])] } )
            else:
                temp.append( { 'candidate_id': c_i['candidate_id'], 'utterance': c_i['utterance'] ,'idx': 0 })
        cand_per_story.append(temp)
    return cand_per_story

def find_similar_candidate(cand_idx, ref):
    scores = []
    for cand, idx in cand_idx.items():
        token_cand = tokenize(cand)
        score = 0
        for pos, word in enumerate(ref):
            if pos < len(token_cand):
                if word == token_cand[pos]:
                    score += 1
        scores.append([ idx, score])
    scores = sorted(scores, key=lambda tup: tup[1], reverse=True)
    if scores!=[]:
         max_score_idx = scores[0][0] if scores[0][1]!=0 else 0
    else:
        max_score_idx = 0
    return max_score_idx

def build_vocab(data, candidates=[]):
    if candidates!=[]:
        vocab = reduce(lambda x, y: x | y, (set(list(chain.from_iterable(d['utter_list'])) + d['q'] ) for d in data))
        vocab |= reduce(lambda x,y: x|y, (set(candidate) for candidate in candidates) )
        vocab = sorted(vocab)
    else:
        vocab = []
        for d in data:
            s, q, a = d['utter_list'], d['q'], d['a']
            c = [ x['utterance'] for x in d['cand_list'] ]
            words = set(list(chain.from_iterable(s)) + q + a + list(chain.from_iterable(c)))
            for word in words:
                if word not in vocab:
                    vocab.append(word)
    return vocab

def data_information(data,FLAGS, candidates=[],officialtestfile=[]):
    dataset=[]

    for i in data:    
        dataset+=i
        
    data=dataset
    vocab=set()
    for i in data:
        for j in i['utterances']:
            vocab=vocab | set(j)
    vocab = list(vocab)

    
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    max_story_size = max(map(len, (d['utterances'] for d in data) ))
    mean_story_size = int(np.mean([ len(d['utterances']) for d in data ]))
    sentence_size = max(map(len, chain.from_iterable(d['utterances'] for d in data)))
    
    return vocab, word_idx, max_story_size, mean_story_size, sentence_size

def tokenize(sent):
    #sent=sent.lower()
    #return sent.split(' ')
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    '''
    sent=sent.lower()
    STOP_WORDS=["'s",",","-",".","<",">","a","of","to"]
    result= [token.text for token in nlp(sent) if token.text not in STOP_WORDS]
   
#    STOP_WORDS=set(["a","an","the"])
#    sent=sent.lower()
#    if sent=='<silence>':
#        return [sent]
#    result=[x.strip() for x in re.split('([^A-Za-z_0-9#]+)?', sent) if x.strip() and x.strip() not in STOP_WORDS]
#    if not result:
#        result=['<silence>']
#    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
#        result=result[:-1]
    return result

def vectorize_data(data, word_idx, sentence_size, batch_size, memory_size, cand_idx,FLAGS):
    S, A = [], []
    for i, d in enumerate(data):
        story, answer = d['utterances'], d['goal']
        ss = []
        for _, sentence in enumerate(story, 1):
            ww = []
            ls = max(0, sentence_size - len(sentence))
            for w in sentence:
               
                if w in word_idx:        
                    ww.append(word_idx[w])
                else:
                    print(w)
      
            ss.append(ww+[0] * ls)   
            
        ss = ss[::-1][:memory_size][::-1]
        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)         


        if len(answer):
            y = [cand_idx[0][answer[-1]['area']],cand_idx[1][answer[-1]['food']],cand_idx[2][answer[-1]['price range']]]
        else:
            y = 0

        S.append(ss)
        A.append(y)
    return S,  A

def vectorize_candidates(cand_idx, idx_cand, word_idx, sentence_size):
    C=[]
    cand_idx_temp = sorted(cand_idx.items(), key=operator.itemgetter(1))
    counter = 0
    for candidate, idx in cand_idx_temp:
        assert idx == cand_idx[candidate] 
        assert idx == counter       
        token_candidate = tokenize(candidate)
        user_utter = True if '$' in token_candidate else False
        if user_utter:
            token_candidate = token_candidate[1:]
            del cand_idx[candidate]
            cand_idx[" ".join(token_candidate)] = idx
        lc=max(0,sentence_size-len(token_candidate))
        if user_utter:
            #print candidate
            C.append([0 for w in token_candidate] + [0] * lc)
        else:
            C.append([word_idx[w] if w in word_idx else 0 for w in token_candidate] + [0] * lc)
        counter += 1
    idx_cand = dict((i, c) for c, i in cand_idx.items()) if cand_idx!=[] else []
    return C, cand_idx, idx_cand

def get_pred_10cands(candsInfo, pred_pro, dialogID=[], rm_unk_sent=False, data=[], idx_cand=[], testInfo=[]):
    preds_only_ten = []
    preds_cand_rank = [] # { 'dialog_id': '', 'lst_candidate_id': [ {'candidate_id': '', 'rank': ''}, ...]}
    for i in range(len(pred_pro)):
        cands = [ x['idx'] for x in candsInfo[i]]
        pred_pro_cands = pred_pro[i][cands]
        if rm_unk_sent:
            for idx, cand in enumerate(cands):
                if cand == 0:
                    pred_pro_cands[idx] = -100   
        pred_ans_idx = np.argmax(pred_pro_cands)
        preds_only_ten.append(cands[pred_ans_idx])
        
        lst_candidate_id = []
        dialog_ranking = {'dialog_id': dialogID[i]}
        sort_index = np.argsort( np.array(pred_pro_cands) )
        sort_value = np.sort( np.array(pred_pro_cands) )
        sort_index = sort_index[::-1]
        sort_value = sort_value[::-1]
        for ii in range(len(sort_index)):
            lst_candidate_id.append( {'candidate_id': candsInfo[i][sort_index[ii]]['candidate_id'], 'rank': str(ii+1) } )

        # Postprocess, deal with addition slot, find the most recent entity in dialog information
        if data!=[] and idx_cand!=[] and testInfo!=[]:
            if sort_value[0] == sort_value[1]:
                c = data[i]['cand_list']
                highest_template =  idx_cand[ cands[sort_index[0]] ]
                for c_i in c:
                    strdiff = ' '.join(c_i['utterance']).replace(highest_template,"").replace(" ","")
                    possKey = 'R_'+strdiff[3:-3]
                    if highest_template in ' '.join(c_i['utterance']) and \
                            possKey in testInfo[i].keys() and \
                            str(len(testInfo[i][possKey])) in c_i['utterance'][-1]:
                        highest_cand_id = c_i['candidate_id']
                        for lst_id in lst_candidate_id:
                            if lst_id['candidate_id']==highest_cand_id:
                                temp = lst_id['rank']
                                lst_id['rank'] = '1'
                                break
                        for lst_id2 in lst_candidate_id:
                            if lst_id2['rank']=='1' and lst_id2['candidate_id']!=highest_cand_id:
                                lst_id2['rank']=temp
                                break
                        #print 'Choose:', c_i['utterance']
                        break
        dialog_ranking['lst_candidate_id'] = lst_candidate_id
        preds_cand_rank.append(dialog_ranking)
    return preds_only_ten, preds_cand_rank

def get_type_dict(kb_path): 
    type_dict = {'R_restaurant':[]}
    fd = open(kb_path, 'rb')
    for line in fd:
        x = line.split('\t')[0].split(' ')
        rest_name = x[1]
        entity = x[2]
        entity_value = line.split('\t')[1].replace('\n','')
        if rest_name not in type_dict['R_restaurant']:
            type_dict['R_restaurant'].append(rest_name)
        if entity not in type_dict.keys():
            type_dict[entity] = []
        if entity_value not in type_dict[entity]:
            type_dict[entity].append(entity_value)
    return type_dict

def combine_SQ(SQ):
    newS = []
    for i in range(len(SQ)):
        newSQ = SQ[i][0]+[SQ[i][1]]
        newSQ = [' '.join(x) for x in newSQ]
        newS.append(newSQ)
    return np.array(newS)

def shuffle_array(shuffle_array_list, shuffle_idx):
    for shuffle_array in shuffle_array_list:
        shuffle_array = shuffle_array[shuffle_idx]

def batch_evaluate(model, S, A, n_data, eval_batch):
    preds = np.empty((0,len(A[0])),dtype=int)
    
    preds_cui = np.empty((0,1),dtype=int)
    preds_loc = np.empty((0,1),dtype=int)
    preds_pri = np.empty((0,1),dtype=int)
    preds_prob = []
    losses = 0.0
    total_batch = int(len(S) / eval_batch)
    jj=int(len(S)!=total_batch*eval_batch) 
    batches=[(eval_batch * i,eval_batch * (i + 1)) if eval_batch * (i + 1) <= len(S) else (eval_batch * i,len(S)) for i in range(total_batch+jj)]

    prog_bar = tqdm(batches)
    for start, end in prog_bar:
        end = start + eval_batch
        s = S[start:end]
        if A!=[]:
            a = A[start:end]
            pred, loss,pred_cui,pred_loc,pred_pri = model.predict(s,  a)
            losses += loss
        else:
            pred = model.predict(s)  
        preds_prob += list(model.predict_proba(s))
        preds =np.concatenate((preds,pred),0) 
        
        preds_cui =np.concatenate((preds_cui,pred_cui[:,None]),0) 
        preds_loc =np.concatenate((preds_loc,pred_loc[:,None]),0) 
        preds_pri =np.concatenate((preds_pri,pred_pri[:,None]),0) 
    return preds, preds_prob, losses/(n_data/eval_batch),preds_cui,preds_loc,preds_pri
