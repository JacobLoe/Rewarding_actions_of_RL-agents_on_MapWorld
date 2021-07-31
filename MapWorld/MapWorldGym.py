from MapWorld.maps import ADEMap
from MapWorld.mapworld import MapWorldWrapper
import numpy as np
from gym import Env
import cv2
from os import path
import json
from sklearn.metrics.pairwise import euclidean_distances


# TODO add option for turning on "dynamic" actions, the agent can only choose actions that are actually available

class MapWorldGym(Env):

    def __init__(self, n=4, m=4, n_rooms=10, room_types=2, room_repetitions=2,
                 ade_path='../../data/ADE20K_2021_17_01/images/ADE/training/',
                 image_resolution=(360, 360),
                 captions="./localized_narratives/ade20k_train_captions.json"):
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

        # keep track of the total steps taken and the return
        self.model_return = 0
        self.model_steps = 0

        self.image_resolution = image_resolution
        # maps the actions actions input into MapWorldGym to actions interpretable by MapWorld
        self.total_available_actions = {0: 'north', 1: 'east', 2: 'south', 3: 'west', 4: 'select_room'}

        ##########################################################
        # The variables defined in the init are placeholders and are only there function is explained here
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
        self.state = []

    def reset(self):
        """
        resets the environment to its initial values
        samples a new map, generates a question from the map
        :return: list, returns the current room, question and available actions
        """
        self.done = False
        # initialise a new MapWorld object with the parameters set in the class init
        ade_map = ADEMap(self.n, self.m, self.n_rooms, (self.room_types, self.room_repetitions))
        self.mw = MapWorldWrapper(ade_map, image_prefix=self.ade_path)

        initial_state = self.mw.initial_state   # the initial state is only needed during the reset

        self.target_room, self.target_room_name = self.get_caption_for_image(self.mw.target_room)
        self.target_room_path = self.mw.target_room

        self.current_room_name = path.relpath(initial_state[0], self.ade_path)
        self.current_room = self.load_image(initial_state[0], self.image_resolution)
        self.current_room_path = initial_state[0]

        # append action 'select_room' directions returned by MapWorld
        self.directions = initial_state[1] + f' or {self.total_available_actions[4]}.'
        # remove prepending fluff, actions are separated by ","
        self.available_actions = self.directions[12:].split()

        # concatenate the target caption and directions

        # keep track of the total steps taken and the return
        self.model_return = 0
        self.model_steps = 0

        # con
        self.text_state = self.target_room + ' ' + self.directions

        # return the initial state
        self.state = {'current_room': self.current_room,
                      'text_state': self.text_state}
        return self.state

    def step(self, action_index):
        """
        Take one step in the environment
        :param action_index: int, index corresponds to one of the actions in self.total_available_actions
        :return: list, contains the state, reward and signal if the game is done
        """
        # map the chosen action index to the action string
        action = self.total_available_actions[action_index]
        if action == self.total_available_actions[4]:
            reward = self.select_room()
        else:
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
        else:
            self.room_found = 0

        reward = self.reward_room_selection()

        # Terminate the game
        self.done = True

        self.state = {'current_room': self.current_room,
                      'text_state': self.directions}
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
            # TODO maybe make step reward linear increasing. Early steps are cheap, later costly
            reward = -1.0 * self.model_steps    # linear increasing reward
            # reward = -10.0 / (1 + np.exp(-self.model_steps + 5)) # reward increasing with sigmoid
            state = self.mw.upd(action)

            self.current_room_name = path.relpath(state[0], self.ade_path)
            self.current_room = self.load_image(state[0], self.image_resolution)
            self.directions = state[1] + f' or {self.total_available_actions[4]}.'
        else:
            # TODO not sure what the correct reward here would be for taking an unavailable action
            # TODO maybe stop penalizing wrong actions
            reward = 0.0

        self.state = {'current_room': self.current_room,
                      'text_state': self.directions}

        return reward

    def reward_room_selection(self):
        """
        Compute the euclidean distance between the feature vectors of two images.

        Returns:
                a float, the reward for selecting a room
        """
        # TODO how to cite ?

        feature0_path = self.target_room_path[:-4] + '.npy'
        feature1_path = self.current_room_path[:-4] + '.npy'

        feature0 = np.load(feature0_path)
        feature1 = np.load(feature1_path)

        # distances follow a gaussian distribution
        distance = euclidean_distances(feature0, feature1)[0]

        # TODO check which normalization to use
        # normalize distance, subtract mean, divide by maximum
        # additionally the sign is switched to map the maximum value to the maximum reward
        normalized_distance = -(distance-16)/31     # range -0.5 to 0.5, mean is 0

        reward = 2000.0 * normalized_distance     # range -1000 to 1000, mean is 0

        return reward

    def get_caption_for_image(self, image_path):
        """
        Extracts a caption for an image to pose as the room to be found on the map
        :param image_path: string
        :return: the captions and the name/category of the target room as a string
        """
        target_room = path.relpath(image_path, self.ade_path)

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
        # TODO make resizing dependent on aspect ratio of source to prevent distortions
        image = cv2.imread(image_path)
        if len(np.shape(image)) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, image_resolution)
        image = np.array(image)

        return image

    def render(self, mode='human'):
        # FIXME probably not really necessary
        pass
