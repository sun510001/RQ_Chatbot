from reverse import reverse_valence
from retrieve import retrieveCommonSense
from rank import rankContext, getRoberta
import sys
import pandas as pd
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU' # if the cpu in your pc is AMD, use this.

# roberta = getRoberta()
#
# utterance = sys.argv[1]
# rov = reverse_valence(utterance).capitalize()
# op = retrieveCommonSense(utterance)
# commonsense, extra = op[0], op[1]
# mostincongruent = rankContext(roberta,rov,commonsense,extra)
# sarcasm = rov + ' '+ mostincongruent
# print(sarcasm)


def sarc_gen(utterance, path):
    roberta = getRoberta()
    all = []
    temp_op = ''
    for i in range(len(utterance)):
        print("robot_res -->", i, utterance[i])
        try:
            rov = reverse_valence(utterance[i]).capitalize()
            op = retrieveCommonSense(utterance[i], path)
            length_op = len(op)
            if length_op >= 2:
                commonsense, extra = op[0], op[1]
            elif length_op == 1:
                commonsense = op[0]
                extra = ''
            else:
                all.append('')
                continue


            if commonsense != temp_op:
                temp_op = commonsense
                most_incongruent = rankContext(roberta, rov, commonsense, extra)
                if most_incongruent != '':
                    sarcasm_show = "sg_res --> {} {} | commonsense:{}; extra:{}\n".format(rov, most_incongruent, commonsense, extra)
                    sarcasm = "{} {}".format(rov, most_incongruent)
                    print(sarcasm_show)
                else:
                    sarcasm = ''
            else:
                sarcasm = ''
        except:
            sarcasm = ''
            pass
        all.append(sarcasm)
    return all


if __name__ == '__main__':
    # utt_list = ["the week started with missing bus and being stranded.", "banks give money to rich people.",
    #             "it is difficult to explain taxation to republicans.", "stressed about getting failed in exams.",
    #             "annoyed by how a white wall is called a painting.", "hate when parents question about money.",
    #             "this stuff is unbelievable."]
    # # utt_list = ["It is difficult to be sad"]
    # utt_list = ["Being stuck in airport is damn boring.", "banks give money to rich people.",
    #                 "it is difficult to explain taxation to republicans.",]
    # utt_list = ["Tell me why that doesn't change the fact that you were in"]

    """for evaluation"""
    # df = pd.read_csv('../dialogue_for_train_after_cla.csv')
    # df = df.dropna(subset=['robot_res'])
    # # df = df[:154]
    # utt_list = df['robot_res'].values.tolist()
    # conda_path = '/home/aquamarine/sunqifan/anaconda3/envs/r_cla/bin/python3.6'
    #
    # out_list = sarc_gen(utt_list, conda_path)
    # df['sg_res'] = out_list
    # df.to_csv('../dialogue_for_train_after_sg.csv', index=False)

    """for chatbot"""
    input = list(sys.argv)
    conda_path = input[1]
    out_list = sarc_gen([input[2]], conda_path)
    print(out_list[0])
    with open("chatbot_out_sarc.txt", 'w') as f:
        f.write(out_list[0])