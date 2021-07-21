import jsonlines
import argparse
import json
import glob
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default='ade20k_train_captions.jsonl', help="")
    parser.add_argument("--out_path", default='ade20k_train_captions.json', help="")
    parser.add_argument('--ADE20K', default='../../../data/ADE20K_2021_17_01/images/ADE/training',
                        help='')
    args = parser.parse_args()

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

    print(len(path))

    op = os.path.join(os.path.split(args.ADE20K)[0], 'no_caption')

    if not os.path.isdir(op):
        os.makedirs(op)

    i = 0
    for p in path:
        im = os.path.split(p)[1]
        # print(im)
        try:
            c = g[im.strip('.jpg')]
        except:
            i += 1
            nc_path = os.path.join(op, im)
            os.rename(p, nc_path)
    print(f'{i} images have been moved to {op}')
