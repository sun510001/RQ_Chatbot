# -*- coding:utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")


def literal_generation_for_evaluation(one_dialogue):
    """generate literal response refer to the previous contexts"""
    count = 0
    for each in one_dialogue[-3:-1]:
        each = re.sub(r'@USER *|<URL>', '', each)
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(each + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if count > 0 else new_user_input_ids
        count += 1
        chat_history_ids = bot_input_ids
        print("H_C: {}".format(each))
        # bot_input_ids = new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    # chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    chat_history_ids = model.generate(
        bot_input_ids,
        do_sample=True,
        max_length=1000,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # pretty print last ouput tokens from bot
    human_response = one_dialogue[-1]
    human_response = re.sub(r'@USER *', '', human_response)
    print("H_R: {}".format(human_response))
    robot_res = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("R_R: {}".format(robot_res))
    print()
    return one_dialogue + [robot_res]


if __name__ == '__main__':
    for step in range(100):
        input_text = input(">> User:")
        if input_text != "stop":
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
            # bot_input_ids = new_user_input_ids

            # generated a response while limiting the total chat history to 1000 tokens,
            # chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            chat_history_ids = model.generate(
                bot_input_ids,
                do_sample=True,
                max_length=1000,
                top_k=50,
                top_p=0.95,
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

            # pretty print last ouput tokens from bot
            print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        else:
            input_text = input(">> input again:")
            new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

            # append the new user input tokens to the chat history
            # bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
            bot_input_ids = new_user_input_ids

            # generated a response while limiting the total chat history to 1000 tokens,
            # chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            chat_history_ids = model.generate(
                bot_input_ids,
                do_sample=True,
                max_length=1000,
                top_k=50,
                top_p=0.95,
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

            # pretty print last ouput tokens from bot
            print("DialoGPT: {}".format(
                tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))