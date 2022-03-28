from PIL import Image
import torch
from torch import nn
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms
import glob
import os
import numpy as np
import pickle
import random
import plotly.express as px
import pandas as pd


def create_histogram(data, title, plot_path='', save_plot=True, save_html=False):
    """

    Args:
        save_html:
        data:
        title:
        plot_path:
        save_plot:
    """

    df = pd.DataFrame(data)
    fig = px.histogram(df, title=title)
    fig.update_xaxes(title_text='normalized distance', showgrid=False, linecolor="#BCCCDC")
    fig.update_yaxes(showgrid=False, linecolor="#BCCCDC")
    fig.update_layout(plot_bgcolor='#FFF')
    if save_plot:
        fig.write_image(plot_path, scale=2.0)
        if save_html:
            html_path = plot_path[:-4] + '.html'
            fig.write_html(html_path)
    else:
        fig.show()


def load_inception(device):
    """
    Loads the InceptionV3 image classification model with pytorch.
    The last two layers (dropout and fully-connected) are removed.
    Returns:
            pytorch InceptionV3 model
    """
    inception = models.inception_v3(pretrained=True, aux_logits=False)
    inception = nn.Sequential(*list(inception.children())[:-2]).to(device)
    inception.eval()
    return inception


def extract_features(model, frame, device):
    """
    Extracts the features
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

    # TODO move code into main function that can be executed by an top-level script
    '''
    The script loads all .jpg-images from the categories used by MapWorld and extracts features for each.
    The features are saved as .npy-files in the same folders. 
    Additionally the euclidean distances between a random sample of 1000 features and all other features are computed.
    A histogram showing the distribution of the distances is created.
    '''
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

    base_path = 'ADE20K_2021_17_01/images/ADE/training/'

    image_path = []
    numpy_path = []

    for p in _target_cats:

        pa = os.path.join(base_path, p, '**/*.jpg')
        image_path.extend(glob.glob(pa, recursive=True))

        pa = os.path.join(base_path, p, '**/*.npy')
        numpy_path.extend(glob.glob(pa, recursive=True))

    print('number of target_cats', len(image_path), len(numpy_path))

    for p in _distractor_cats:

        pa = os.path.join(base_path, p, '**/*.jpg')
        image_path.extend(glob.glob(pa, recursive=True))

        pa = os.path.join(base_path, p, '**/*.npy')
        numpy_path.extend(glob.glob(pa, recursive=True))

    print('number of target_cats + distractor_cats', len(image_path), len(numpy_path))

    for p in _outdoor_cats:

        pa = os.path.join(base_path, p, '**/*.jpg')
        image_path.extend(glob.glob(pa, recursive=True))

        pa = os.path.join(base_path, p, '**/*.npy')
        numpy_path.extend(glob.glob(pa, recursive=True))

    print('number of target_cats + distractor_cats + outdoor_cats', len(image_path), len(numpy_path))

    # compute the distances between the features to get the mean and max distance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = load_inception(device)

    pickled_dists = 'utils/distances.pkl'
    sample_size = 1000#7917

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
    elif not os.path.isfile(pickled_dists):
        print('load features')
        features = []
        for fp in tqdm(numpy_path):
            features.append(np.load(fp))

        print('features shape', np.shape(features))
        # get a random sample
        features_sample = random.sample(features, sample_size)
        print('features_sample shape', np.shape(features_sample))

        fs = [(f0, f1) for f0 in features for f1 in features_sample]
        dists = []
        # for f0 in tqdm(features):
        #     d = [euclidean_distances(f0, f)[0] for f in features]
        #     dists.extend(d)
        # shifts the distance distribution. Should be between -0.5 and 0.5 to keep the distribution between
        # positive values shift the distribution to reward higher/more positive
        # negative values shift the distribution to reward lower/more negative
        # with the default distribution 50% of the reward is positive / max/min/mean 0.5/-0.5/0
        # with shift_factor = 0.5 100% of the reward is positive max/min/mean 1/0/0.5
        # with shift_factor = 0.25 99% of the reward is positive max/min/mean 0.75/-0.25/0.25
        shift_factor = 0
        for f0, f1 in tqdm(fs):
            # distribution between -0.5 and 0.5
            norm_dist = (euclidean_distances(f0, f1)[0])/1#30.135202
            #
            dists.append((norm_dist+shift_factor))
        # clipped_dists = np.clip(dists, 0, 1)
        # with open(pickled_dists, 'wb') as f:
        #     pickle.dump(dists, f)
    else:
        print('loading pickled distances')
        with open(pickled_dists, 'rb') as f:
            dists = pickle.load(f)
        print('dists', type(dists), np.shape(dists))

    print('min, max, mean, std', np.min(dists), np.max(dists), np.mean(dists), np.std(dists))

    dists = (np.array(dists)-15.956363)/30.135202

    title = f'Distances for {sample_size} randomly chosen images to every other image. ' \
            f'<br>Mean: {np.round(np.mean(dists), decimals=4)}, ' \
            f'<br>Max: {np.round(np.max(dists), decimals=4)}, ' \
            f'Min: {np.round(np.min(dists), decimals=4)}, '

    create_histogram(dists, title, plot_path='utils/dists.png', save_plot=True)
