
from score import do_compute_score, do_load_json_result
import json
#path_to_data=['../data/dataset-E2E-goal-oriented-test-v1.0/tst2/dialog-task2REFINE-kb2_atmosphere-distr0.5-tst1000.answers.json','../data/dataset-E2E-goal-oriented-test-v1.0/tst2/dialog-task2REFINE-kb2_atmosphere-distr0.5-tst1000.json']
path_to_data=['../data/dataset-E2E-goal-oriented-test-v1.0/tst4/dialog-task1API-kb2_atmosphere_restrictions-distr0.5-tst1000.answers.json','../data/dataset-E2E-goal-oriented-test-v1.0/tst4/dialog-task1API-kb2_atmosphere_restrictions-distr0.5-tst1000.json']

fd = open(path_to_data[1], 'rb')
json_data = json.load(fd)
fd.close()
answer_dict=do_load_json_result(path_to_data[0],1)

cand={}

for i in range(len(json_data)):
    for c in json_data[i]['candidates']:
        cand[c['candidate_id']]=c['utterance']
                
testdata=json_data
for i in range(len(json_data)):
    testdata[i]['answer']={'candidate_id':answer_dict[testdata[i]['dialog_id']][0] ,'utterance':cand[answer_dict[testdata[i]['dialog_id']][0]]}

js = json.dumps(testdata)

#fp = open('dialog-task2REFINE-kb2_atmosphere-test2.json', 'a')
fp = open('dialog-task1API-kb2_atmosphere_restrictions-test4.json', 'a')
fp.write(js)
fp.close()