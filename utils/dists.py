from PIL import Image
import torch
from torch import nn
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms
import glob
import os
from plots import create_histogram
import numpy as np
import pickle


def extract_features(model, frame, device):
    """

    Args:
        model:
        frame:
        device:

    Returns:

    """
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    processed_frame = preprocess(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model(processed_frame).squeeze().cpu().detach().numpy()
        feature = feature.reshape(1, -1)
    return feature


if __name__ == '__main__':

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

    base_path = '../../data/ADE20K_2021_17_01/images/ADE/training/'

    image_path = []
    numpy_path = []

    for p in _target_cats:

        pa = os.path.join(base_path, p, '**/*.jpg')
        image_path.extend(glob.glob(pa, recursive=True))

        pa = os.path.join(base_path, p, '**/*.npy')
        numpy_path.extend(glob.glob(pa, recursive=True))

    print('target_cats', len(image_path), len(numpy_path))

    for p in _distractor_cats:

        pa = os.path.join(base_path, p, '**/*.jpg')
        image_path.extend(glob.glob(pa, recursive=True))

        pa = os.path.join(base_path, p, '**/*.npy')
        numpy_path.extend(glob.glob(pa, recursive=True))

    print('distractor_cats', len(image_path), len(numpy_path))

    for p in _outdoor_cats:

        pa = os.path.join(base_path, p, '**/*.jpg')
        image_path.extend(glob.glob(pa, recursive=True))

        pa = os.path.join(base_path, p, '**/*.npy')
        numpy_path.extend(glob.glob(pa, recursive=True))

    print('outdoor_cats', len(image_path), len(numpy_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = models.inception_v3(pretrained=True, aux_logits=False)
    # print(inception, '\n')
    inception = nn.Sequential(*list(inception.children())[:-2]).to(device)
    # print(inception)
    inception.eval()

    if not len(image_path) == len(numpy_path):
        dists = []
        i = Image.open(image_path[0])
        f0 = extract_features(inception, i, device)
        print('load images')
        for p in tqdm(image_path):
            i = Image.open(p)
            if len(np.shape(i)) != 3:
                i = i.convert('RGB')

            f = extract_features(inception, i, device)
            fp = p[:-4] + '.npy'
            np.save(fp, f)

            d = euclidean_distances(f0, f)
            dists.append(d[0])
    else:
        print('load features')
        features = []
        for fp in tqdm(numpy_path):
            features.append(np.load(fp))

        # TODO save dists with pickle
        fs = [(f0, f1) for f0 in features for f1 in features]

        print('saved features')
        dists = []
        for f0, f1 in tqdm(fs):
            dists.append(euclidean_distances(f0, f1))
        with open('utils/distances.pkl', 'wb') as f:
            pickle.dump(dists, f)

    print(len(dists))

    print('min, max, mean', np.min(dists), np.max(dists), np.mean(dists))

    create_histogram(dists, 'dists')
