from MapWorld.maps import ADEMap
from MapWorld.mapworld import MapWorldWrapper
import numpy as np
from gym import Env, spaces
import cv2
from PIL import Image
from os import path
import json
from sklearn.metrics.pairwise import euclidean_distances

import time


class MapWorldGym(Env):

    def __init__(self, n=4, m=4, n_rooms=10, room_types=2, room_repetitions=2,
                 ade_path='../../data/ADE20K_2021_17_01/images/ADE/training/',
                 image_resolution=(360, 360),
                 captions="./localized_narratives/ade20k_train_captions.json",
                 reward_constant_step=-10.0,
                 reward_linear_step=-0.6666,
                 reward_logistic_step=-10.0,
                 reward_wrong_action=0.0,
                 reward_room_selection=2000.0,
                 penalty_room_selection=-1000.0,
                 reward_selection_by_distance='True',
                 reward_step_function='linear',
                 images_returned_as_array='True'):
        # the dimensions of the map
        self.n = n
        self.m = m
        # the number of rooms on the map
        self.n_rooms = n_rooms
        # TODO add explanation for room types and repetitions (what do they mean for the maps)
        self.room_types = room_types
        self.room_repetitions = room_repetitions

        with open(captions, 'r') as f:  # max length of caption 213
            self.dict_captions = json.load(f)

        #
        self.ade_path = ade_path

        # done is False as long as an episode is running
        # when an episode terminates room_found is turned to 1 when the correct was selected
        self.done = False
        self.room_found = 0

        # TODO remove, are not needed
        # keep track of the total steps taken and the return
        self.model_return = 0
        self.model_steps = 0

        self.image_resolution = image_resolution

        # maps the actions actions input into MapWorldGym to actions that are interpretable by MapWorld
        self.actions = {0: 'north', 1: 'east', 2: 'south', 3: 'west', 4: 'select_room'}
        self.action_space = spaces.Discrete(len(self.actions))

        # TODO add type checks for reward function
        # define the rewards (and penalties) for taking actions
        self.reward_constant_step = reward_constant_step
        self.reward_linear_step = reward_linear_step
        self.reward_logistic_step = reward_logistic_step
        self.reward_room_selection = reward_room_selection
        self.reward_wrong_action = reward_wrong_action
        self.penalty_room_selection = penalty_room_selection

        # check which reward function is to be used for takings steps and if it is valid
        rsf = {'constant': 0, 'linear': 1, 'logistic': 2}
        if reward_step_function not in rsf.keys():
            raise Exception(f'The reward function for a step has to be one of {rsf}. Was "{reward_step_function}"')
        self.reward_step_function = rsf[reward_step_function]

        self.reward_selection_by_distance = True if reward_selection_by_distance == 'True' else False

        self.images_returned_as_array = True if images_returned_as_array == 'True' else False

        ##########################################################
        # The variables defined in the init are placeholders and are only their function is explained here
        # They only have proper values after calling the reset function

        # current_room is represented by an image as a numpy array
        # current_room_name the contains the file name of the image
        # current_room_path contains the the path to image
        self.current_room = np.array([])
        self.current_room_name = ''
        self.current_room_path = ''

        # the target is represented by a caption generated from the image from a randomly sampled room
        # target_room_name the filename of the image
        # target_room_path contains the the path to image
        self.target_room = ''
        self.target_room_name = ''
        self.target_room_path = ''

        # directions contains the available actions in the current room as a string (returned as part of the state)
        # 'select_room' is always added at the end
        # available_actions contains them as a list for internal use of the environment
        self.directions = ''
        self.available_actions = []

        #
        self.text_state = ''

        # the state consists of: current room as numpy ndarray of shape (, , 3),
        # target room question(caption>) as string,
        # available actions as list of string
        self.state = {}

    def reset(self):
        """
        resets the environment to its initial values
        samples a new map, generates a question from the map
        :return: list, returns the current room, question and available actions
        """
        self.done = False
        # keep track of the total steps taken and the return
        self.model_return = 0
        self.model_steps = 0

        # initialise a new MapWorld object with the parameters set in the class init
        ade_map = ADEMap(self.n, self.m, self.n_rooms, (self.room_types, self.room_repetitions))
        self.mw = MapWorldWrapper(ade_map, image_prefix=self.ade_path)

        initial_state = self.mw.initial_state

        #
        self.target_room, self.target_room_name = self.get_caption_for_image(self.mw.target_room)
        self.target_room_path = self.mw.target_room

        #
        self.current_room_name = path.relpath(initial_state[0], self.ade_path)
        self.current_room = self.load_image(initial_state[0], self.image_resolution)
        self.current_room_path = initial_state[0]

        # append action 'select_room' directions returned by MapWorld
        self.directions = initial_state[1] + f' or {self.actions[4]}.'
        # get a searchable list of available actions
        # remove prepending fluff, actions are separated by "," and "or"
        self.available_actions = self.directions[12:].replace('or ', '').strip('.').split()

        # concatenate the target caption and directions
        self.text_state = self.target_room + ' ' + self.directions

        # return the initial state as a dictionary
        self.state = {'current_room': self.current_room,
                      'text_state': self.text_state}
        return self.state

    def step(self, action_index):
        """
        Take one step in the environment. An index is mapped to the string representing the actions in MapWorld
        :param action_index: int, index corresponds to one of the actions in self.actions
        :return: list, contains the state, reward and signal if the game is done
        """
        # map the chosen action index to the action string
        if action_index == 4:
            reward = self.select_room()
        else:
            action = self.actions[action_index]
            reward = self.move(action)

        self.model_return += reward
        self.model_steps += 1

        return self.state, reward, self.done, self.room_found

    def select_room(self):
        """
        Select_room ends the episode
        Returns: a float, the reward for the action
        """
        if self.current_room_name == self.target_room_name:
            self.room_found = 1
            reward = self.reward_room_selection
        else:
            self.room_found = 0
            reward = self.penalty_room_selection

        # reward the room selection based on the similarity between target and current room
        # overrides any rewards set previously
        if self.reward_selection_by_distance:
            reward = self.get_reward_from_distance()

        # Terminate the game
        self.done = True

        self.state = {'current_room': self.current_room,
                      'text_state': self.text_state}
        return reward

    def move(self, action):
        """
        Moves in the map according to the action. If the action is not in the available actions,
        the agent stays in the current room and the current state is returned again.
        Args:
            action: string, one from north, south, west, east

        Returns: a float, the reward for the action
        """
        if action in self.available_actions:
            if self.reward_step_function == 0:
                # constant reward
                reward = self.reward_constant_step
            elif self.reward_step_function == 1:
                # linear increasing reward
                reward = self.reward_linear_step * self.model_steps
            elif self.reward_step_function == 2:
                # reward increasing with logistic function
                # TODO explain the boundaries for sigmoid
                reward = self.reward_logistic_step / (1 + np.exp(-self.model_steps + 11))

            # input the action to mapworld and update the state accordingly
            state = self.mw.upd(action)

            self.current_room_name = path.relpath(state[0], self.ade_path)
            self.current_room = self.load_image(state[0], self.image_resolution)

            self.directions = state[1] + f' or {self.actions[4]}.'
            self.available_actions = self.directions[12:].replace('or', '').strip('.').split()

            self.text_state = self.target_room + ' ' + self.directions
        else:
            reward = self.reward_wrong_action

        self.state = {'current_room': self.current_room,
                      'text_state': self.text_state}

        return reward

    def get_reward_from_distance(self):
        """
        Compute the euclidean distance between the feature vectors of two images.

        Returns:
                a float, the reward for selecting a room
        """
        # TODO how to cite hpi work ?

        path_feature_target = self.target_room_path[:-4] + '.npy'
        path_feature_current = self.current_room_path[:-4] + '.npy'

        feature_target = np.load(path_feature_target)
        feature_current = np.load(path_feature_current)

        # distances follow a gaussian distribution
        distance = euclidean_distances(feature_target, feature_current)[0][0]
        # normalize distance, subtract mean, divide by maximum
        # mean and max distance were computed from the distance of all images to other image
        # additionally the sign is switched to map the minimum distance value to the maximum reward
        normalized_distance = -(distance-15.956363)/30.135202     # range -0.5 to 0.5, mean is 0

        # 1000 when the rooms are the exact same,
        reward = self.reward_room_selection * normalized_distance     # range -1000 to 1000, mean is 0
        return reward

    def get_caption_for_image(self, image_path):
        """
        Extracts a caption for an image to pose as the room to be found on the map
        :param image_path: string
        :return: the captions and the name/category of the target room as a string
        """
        target_room = path.relpath(image_path, self.ade_path)

        # use the name of the image as the key for the caption dictionary
        caption_key = path.split(target_room)[1].strip('.jpg')
        target_caption = self.dict_captions[caption_key]

        return target_caption, target_room

    def load_image(self, image_path, image_resolution):
        """
        Loads an image from disk and reshapes it to a given resolution.
        In case the source was in grayscale, it gets transformed to a RGB image
        Args:
            image_path: string
            image_resolution: tuple

        Returns: Numpy array, reshaped image, (height, width, channels)
        """
        if self.images_returned_as_array:
            # return the image as a numpy array
            # TODO make resizing dependent on aspect ratio of source to prevent distortions
            image = cv2.imread(image_path)
            # reshape any grayscale images to rgb
            if len(np.shape(image)) != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = cv2.resize(image, image_resolution)

            image = np.array(image)
        else:
            # return the image as a Pillow object
            image = Image.open(image_path)
            if len(np.shape(image)) != 3:
                image = image.convert('RGB')
        return image
