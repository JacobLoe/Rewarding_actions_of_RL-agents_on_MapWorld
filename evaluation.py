from agents import RandomBaseline
from MapWorld import MapWorldGym
# from im2txt import Captioning
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')
        print(np.shape(new_user_input_ids), type(new_user_input_ids))
        for i in new_user_input_ids:
            print(np.shape(i), type(i))
        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
    # mwg = MapWorldGym()
    # initial_state = mwg.reset()
    # print(np.shape(initial_state))
    # print(initial_state)
    # available_actions = mwg.available_actions
    # num_actions = mwg.num_actions
    # print(num_actions, available_actions)

    # while not mwg.done:
    #     action = available_actions[rb.select_action(len(available_actions))]
    #     s = mwg.step(action)
    #     available_actions = s[0][2]
    #     print(s[0][2], s[1])

    # obj = Captioning("./im2txt/checkpoints/5M_iterations/model.ckpt-5000000", './im2txt/vocab/word_counts.txt')
    #
    # cap = obj.image("./MapWorld/ADE20k_test.jpg")
    #
    # print(cap)
    # print(type(cap))
