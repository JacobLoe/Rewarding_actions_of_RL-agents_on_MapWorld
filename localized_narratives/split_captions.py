from tqdm import tqdm
import argparse
import json
import glob
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default='../../final-pairs/training-pairs/', help="")
    parser.add_argument("--out_path", default='ade20k_train_split_captions.json', help="")

    args = parser.parse_args()

    # get all captions
    p = os.path.join(args.in_path, '*.txt')
    captions_path_list = glob.glob(p)

    split_captions = {}
    for caption in tqdm(captions_path_list):
        with open(caption, 'r') as f:
            lines = [s.strip('\n') for s in f.readlines()]
            image_name = os.path.split(caption)[1].strip('.txt')
            split_captions[image_name] = lines

    with open(args.out_path, 'w') as f:
        json.dump(split_captions, f, sort_keys=True)

