# -*- coding:utf-8 -*-
import pandas as pd
import json
import os, sys
from __init__ import LoadDialogueDataset, PATH_PYTHON
import pandas as pd
from sklearn.model_selection import train_test_split
from sentiment_check import find_sentiment
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import subprocess

def tok_encode_ids(input_text, tokenizer):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    return new_user_input_ids


def model_generate(input_ids, tokenizer, model):
    chat_history_ids = model.generate(
        input_ids,
        do_sample=True,
        max_length=1000,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    return chat_history_ids


class StackList():
    """the stack that can only save three sentences"""
    def __init__(self, input_data):
        self.list_stack = [''] * 3
        self.input_data = input_data

    def clean(self):
        self.list_stack = [''] * 3

    def push(self, input_data):
        self.list_stack = self.list_stack[1:] + [input_data]


def cla_sarc_rq_generator(input_utt_list):
    """situation classification"""
    os.chdir(os.getcwd() + '/Situation_Classification_for_SRL')
    subprocess.Popen([PATH_PYTHON, 'run_predict.py', input_utt_list[0], input_utt_list[1]],
                     stdout = subprocess.PIPE, stderr = subprocess.STDOUT).wait()
    with open("chatbot_out.txt", "r") as f:
        cla_result = f.readline()
    os.chdir(os.getcwd() + '/..')

    out_result = None
    # cla_result = 1
    if cla_result == 2:
        """sarcasm generation"""
        os.chdir(os.getcwd() + '/SarcasmGeneration-ACL2020')
        subprocess.Popen([PATH_PYTHON, 'generate_sarcasm.py', PATH_PYTHON, input_utt_list[2]],
                         stdout = subprocess.PIPE, stderr = subprocess.STDOUT).wait()
        with open("chatbot_out_sarc.txt", 'r') as f:
            out_result = f.readline()
        os.chdir(os.getcwd() + '/..')


    elif cla_result == 1:
        """RQ generation"""
        os.chdir(os.getcwd() + '/RQ_generator')
        subprocess.Popen([PATH_PYTHON, 'run_train_classifier.py', input_utt_list[0], input_utt_list[1], input_utt_list[2]],
                         stdout = subprocess.PIPE, stderr = subprocess.STDOUT).wait()
        with open("chatbot_out_rq.txt", 'r') as f:
            out_result = f.readline()
        os.chdir(os.getcwd() + '/..')

    if not (isinstance(out_result, type(None)) or out_result == ''):
        first_two_list = input_utt_list[:2]
        first_two_list.append(out_result)
    else:
        first_two_list = input_utt_list

    return first_two_list


def literal_generation_for_chatbot(tokenizer, model):
    """generate literal response in a conversation (interactive mode)"""
    cr_stack = StackList('')
    print("\n You can use \"r\" to clean the cache of the chatbot")
    for step in range(100):
        # if the input is refresh, the weight that is going to be input in the model will be cleaned
        input_text = input(">> User:")

        if step > 0 and input_text != 'r':
            # append the new user input tokens to the chat history
            new_user_input_ids = tok_encode_ids(input_text, tokenizer)
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        elif input_text == 'r':
            context_response_stack = ['']*3
            print("Cache cleaned.")
            cr_stack.clean()  # clean the stack
            input_text = input(">> User:")
            new_user_input_ids = tok_encode_ids(input_text, tokenizer)
            bot_input_ids = new_user_input_ids
        else:
            # when step is 0
            new_user_input_ids = tok_encode_ids(input_text, tokenizer)
            bot_input_ids = new_user_input_ids

        cr_stack.push(input_text)  # stack user's utterance

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model_generate(bot_input_ids, tokenizer, model)

        # pretty print last ouput tokens from bot
        decode_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        cr_stack.push(decode_response)
        """start the classification"""
        if '' not in cr_stack.list_stack:
            cr_stack.list_stack = cla_sarc_rq_generator(cr_stack.list_stack)

        print("DialoGPT: {}".format(cr_stack.list_stack[-1]))


if __name__ == '__main__':
    """chatbot literal response generation"""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
    literal_generation_for_chatbot(tokenizer, model)



