# from agents import RandomBaseline
# from MapWorld import MapWorldGym
# from utils.evaluation import eval_rand_baseline, evaluate_model
# import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    text = r"""
    The room is kitchen with a refrigerator, a sink and a dishwasher. Plates and cups are on the desk. The walls of the kitchen are blue.
    """
    questions = ["What color is the room ?", "What objects are in the room ?", "Is there something on the desk ?"]

    for question in questions:
        inputs = tokenizer.encode_plus(question, add_special_tokens=True)
        print(inputs)

        # input_ids = inputs["input_ids"].tolist()[0]

        # # text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        # answer_start_scores, answer_end_scores = model(**inputs, return_dict=False)
        #
        # torch.tensor(answer_end_scores)
        # answer_start = torch.argmax(
        #     answer_start_scores
        # )  # Get the most likely beginning of answer with the argmax of the score
        # answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        #
        # answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        #
        # print(f"Question: {question}")
        # print(f"Answer: {answer}\n")

    # mwg = MapWorldGym()
    # rb = RandomBaseline()
    # model_return, model_steps = evaluate_model(mwg, rb, eval_rand_baseline)
    # print(model_return)
    # print(np.mean(model_return))
    # print('-------------------')
    # print(model_steps)
    # print(np.mean(model_steps))
