from tqdm import tqdm
import numpy as np
import argparse
import json
import glob
import os
import jsonlines

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default='../../final-pairs/training-pairs/', help="")
    parser.add_argument("--out_path", default='ade20k_train_split_captions.json', help="")

    args = parser.parse_args()

    # get all captions
    p = os.path.join(args.in_path, '*.txt')
    captions_path_list = glob.glob(p)

    localized_narratives = {}
    with jsonlines.open('ade20k_train_captions.jsonl', mode='r') as f:
        for line in f:
            image_name = line['image_id']
            caption = line['caption']
            localized_narratives[image_name] = caption

    print('localized narratives')
    c = [len(localized_narratives[k].split()) for k in localized_narratives]
    print(f'Min caption length {np.min(c)}')
    print(f'Max caption length {np.max(c)}')
    print(f'Mean caption length {np.mean(c)}')

    # TODO fix missing captions and wrong captions
    split_captions = {}
    for caption_path in tqdm(captions_path_list):
        with open(caption_path, 'r') as f:
            lines = [s.strip('\n') for s in f.readlines()]
            image_name = os.path.split(caption_path)[1].strip('.txt')
            if len(lines) != 0:
                if len(lines[0].split()) == 236:
                    split_captions[image_name] = ['One person standing']
                else:
                    split_captions[image_name] = lines
            else:
                split_captions[image_name] = localized_narratives[image_name]

    for k in tqdm(split_captions):
        f = len(split_captions[k][0].split())
        if f == 1:
            print(split_captions[k])

    print('split captions')
    c = [len(split_captions[k][0].split()) for k in split_captions]
    print(f'Min caption length {np.min(c)}')
    print(f'Max caption length {np.max(c)}')
    print(f'Mean caption length {np.mean(c)}')

    # with open(args.out_path, 'w') as f:
    #     json.dump(split_captions, f, sort_keys=True)

