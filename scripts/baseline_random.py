
import random
import json


def do_parse_cmdline():

    from optparse import OptionParser
    parser = OptionParser()

    parser.add_option("--input-task-file", dest="inputtaskfile",
                      default="/Users/yangyang/Dialog project/dstc6-goal-oriented-end-to-end-master/data/dataset-E2E-goal-oriented/dialog-task1API-kb1_atmosphere-distr0.5-trn10000.json",
                      help="filename of the task", metavar="FILE")

    parser.add_option("--output_result-file", dest="outputresultfile",
                      default="output-result2.json",
                      help="output file results", metavar="FILE")

    (options, args) = parser.parse_args()

    return options.inputtaskfile, options.outputresultfile


### The dialog format
### [{dialog_id : " ", lst_candidate_id: [{candidate_id: " ", rank: " "}, ...]}]

if __name__ == '__main__':

    # Parsing command line
    inputtaskfile, outputresultfile = do_parse_cmdline()

    fd = open(inputtaskfile, 'rb')
    json_data = json.load(fd)
    fd.close()

    lst_responses = []

    for story in json_data:
        dict_answer_current = {}
        dict_answer_current['dialog_id'] = story['dialog_id']

        lst_candidate_id = []
        for cand in story['candidates']:
            lst_candidate_id.append(cand['candidate_id'])
        random.shuffle(lst_candidate_id)

        lst_candidate_rank = []
        for it in range (0, len(lst_candidate_id)):
            lst_candidate_rank.append({"candidate_id": lst_candidate_id[it], "rank": it+1})

        dict_answer_current['lst_candidate_id'] = lst_candidate_rank
        lst_responses.append(dict_answer_current)

    fd = open(outputresultfile, 'w')
    json.dump(lst_responses, fd)
    fd.close()