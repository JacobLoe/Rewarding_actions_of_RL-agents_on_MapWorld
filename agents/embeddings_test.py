import jsonlines
import numpy as np
import torch
from transformers import AutoTokenizer, BertModel

localized_narratives = {}
with jsonlines.open('localized_narratives/ade20k_train_captions.jsonl', mode='r') as f:
   for line in f:
      image_name = line['image_id']
      caption = line['caption']
      localized_narratives[image_name] = caption

c = [len(localized_narratives[k].split()) for k in localized_narratives]
print(f'Min caption length {np.min(c)}')
print(f'Max caption length {np.max(c)}')
print(f'Mean caption length {np.mean(c)}')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# # TODO use a model which produces embeddings <786
# # TODO get size of the embeddings directly from the model
bert_embeddings = BertModel.from_pretrained('bert-base-uncased')

for k in localized_narratives:
   text = localized_narratives[k]
   break
print('text length: ', len(text.split()))

max_sequence_length = 213

text_tokens = tokenizer.encode_plus(text, padding='max_length',
                                    max_length=max_sequence_length,
                                    add_special_tokens=True)
# print(text_tokens)

text_tokens_tensor = torch.LongTensor(text_tokens['input_ids']).unsqueeze(0)

bert_output = bert_embeddings(text_tokens_tensor)

print('\n')
print(bert_output[0][0].size(), type(bert_output[0][0]))