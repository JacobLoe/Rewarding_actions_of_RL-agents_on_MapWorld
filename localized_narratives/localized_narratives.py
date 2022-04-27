'''
This script goes through the specified categories of the ADE20K dataset and
moves all images without a caption to a new folder.
'''


import jsonlines
import argparse
import json
import glob
import os
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default='localized_narratives/ade20k_train_captions.jsonl', help="")
    parser.add_argument("--out_path", default='localized_narratives/ade20k_train_captions.json', help="")
    parser.add_argument('--ADE20K', default='ADE20K_2021_17_01/images/ADE/training',
                        help='')
    args = parser.parse_args()

    # transform jsonlines file into a json file with only the imagee id and the caption
    localized_narratives = {}
    with jsonlines.open(args.in_path, mode='r') as f:
        for line in f:
            image_name = line['image_id']
            caption = line['caption']
            localized_narratives[image_name] = caption

    with open(args.out_path, 'w') as f:
        json.dump(localized_narratives, f, sort_keys=True)

    with open(args.out_path, 'r') as f:
        g = json.load(f)

    _target_cats = ['home_or_hotel/bathroom', 'home_or_hotel/bedroom', 'home_or_hotel/kitchen',
                    'home_or_hotel/basement', 'home_or_hotel/nursery', 'home_or_hotel/attic', 'home_or_hotel/childs_room',
                    'home_or_hotel/playroom', 'home_or_hotel/dining_room', 'home_or_hotel/home_office',
                    'work_place/staircase', 'home_or_hotel/utility_room', 'home_or_hotel/living_room',
                    'sports_and_leisure/jacuzzi__indoor', 'transportation/doorway__indoor',
                    'sports_and_leisure/locker_room',
                    'shopping_and_dining/wine_cellar__bottle_storage', 'work_place/reading_room',
                    'work_place/waiting_room', 'urban/balcony__interior']

    _distractor_cats = ['home_or_hotel/home_theater', 'work_place/storage_room', 'home_or_hotel/hotel_room',
                        'cultural/music_studio', 'work_place/computer_room', 'urban/street',
                        'urban/yard', 'shopping_and_dining/tearoom', 'cultural/art_studio',
                        'cultural/kindergarden_classroom', 'work_place/sewing_room',
                        'home_or_hotel/shower', 'urban/veranda', 'shopping_and_dining/breakroom',
                        'urban/patio', 'home_or_hotel/garage__indoor',
                        'work_place/restroom__indoor', 'work_place/workroom', 'work_place/corridor',
                        'home_or_hotel/game_room', 'home_or_hotel/poolroom__home', 'shopping_and_dining/cloakroom__room',
                        'home_or_hotel/closet', 'home_or_hotel/parlor', 'transportation/hallway', 'work_place/reception',
                        'transportation/carport__indoor', 'home_or_hotel/hunting_lodge__indoor']

    _outdoor_cats = ['urban/garage__outdoor', 'urban/apartment_building__outdoor',
                     'sports_and_leisure/jacuzzi__outdoor', 'urban/doorway__outdoor',
                     'urban/restroom__outdoor', 'sports_and_leisure/swimming_pool__outdoor',
                     'urban/casino__outdoor', 'urban/kiosk__outdoor',
                     'urban/apse__outdoor', 'urban/carport__outdoor',
                     'urban/flea_market__outdoor', 'urban/chicken_farm__outdoor',
                     'urban/washhouse__outdoor', 'urban/cloister__outdoor',
                     'urban/diner__outdoor', 'urban/kennel__outdoor',
                     'urban/hunting_lodge__outdoor', 'urban/cathedral__outdoor',
                     'urban/newsstand__outdoor', 'urban/parking_garage__outdoor',
                     'urban/convenience_store__outdoor', 'urban/bistro__outdoor',
                     'urban/inn__outdoor', 'urban/library__outdoor']

    base_path = args.ADE20K

    path = []

    for p in _target_cats:

        pa = os.path.join(base_path, p, '**/*.jpg')

        path.extend(glob.glob(pa, recursive=True))

    for p in _distractor_cats:

        pa = os.path.join(base_path, p, '**/*.jpg')

        path.extend(glob.glob(pa, recursive=True))

    for p in _outdoor_cats:

        pa = os.path.join(base_path, p, '**/*.jpg')

        path.extend(glob.glob(pa, recursive=True))

    print(f'The categories include {len(path)} images to be checked for captions')

    op = os.path.join(os.path.split(args.ADE20K)[0], 'no_caption')

    if not os.path.isdir(op):
        os.makedirs(op)

    new_captions = []
    i = 0
    for p in path:
        im = os.path.split(p)[1]
        try:
            c = g[im.strip('.jpg')]
            new_captions.append(c)
        except:
            i += 1
            nc_path = os.path.join(op, im)
            os.rename(p, nc_path)
    print(f'{i} images with no captions have been found and moved to {op}')

    print('len all captions', len(g))
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(g.values())
    print('Vocabulary size all captions', len(vectorizer.vocabulary_))

    c = [len(localized_narratives[k].split()) for k in localized_narratives]
    print(f'Min caption length {np.min(c)}')
    print(f'Max caption length {np.max(c)}')
    print(f'Mean caption length {np.mean(c)}')
    print(f'Median caption length {np.median(c)}')

    for k in localized_narratives.values():
        if len(k.split()) == 5:
            print(k)
            break
    for k in localized_narratives.values():
        if len(k.split()) == 37:
            print(k)
            break

    fig = px.histogram(c, title='Histogram of caption lengths for ADE20K')
    fig.update_xaxes(title_text='Caption length', showgrid=False, linecolor="#BCCCDC")
    fig.update_yaxes(showgrid=False, linecolor="#BCCCDC")
    fig.update_layout(plot_bgcolor='#FFF', showlegend=False)
    fig.write_image('caption_length_ADE20K.png', scale=2.0)

    print('\nlen captions for MapWorld', len(new_captions))
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(new_captions)
    print('Vocabulary size MapWorld captions', len(vectorizer.vocabulary_))

    c = [len(k.split()) for k in new_captions]
    print(f'Min caption length {np.min(c)}')
    print(f'Max caption length {np.max(c)}')
    print(f'Mean caption length {np.mean(c)}')
    print(f'Median caption length {np.median(c)}')

    for k in new_captions:
        if len(k.split()) == 7:
            print(k)
            break
    for k in new_captions:
        if len(k.split()) == 45:
            print(k)
            break

    fig = px.histogram(c, title='Histogram of caption lengths for MapWorld')
    fig.update_xaxes(title_text='Caption length', showgrid=False, linecolor="#BCCCDC")
    fig.update_yaxes(showgrid=False, linecolor="#BCCCDC")
    fig.update_layout(plot_bgcolor='#FFF', showlegend=False)
    fig.write_image('caption_length_MapWorld.png', scale=2.0)
