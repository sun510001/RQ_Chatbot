from reverse import reverse_valence
from retrieve import retrieveCommonSense
from rank import rankContext, getRoberta
import sys
import pandas as pd
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU' # if the cpu in your pc is AMD, use this.

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