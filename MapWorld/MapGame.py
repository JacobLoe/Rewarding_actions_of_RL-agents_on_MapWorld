import os

from MapWorld.maps import ADEMap
from MapWorld.mapworld import MapWorldWrapper
import numpy as np
import gym
import cv2
from os import path
import json


# TODO maybe accelerate game by loading all images into cache before hand
# TODO add option for turning on "dynamic" actions, the agent can only choose actions that are actually available

class MapWorldGym(gym.Env):

    def __init__(self, n=4, m=4, n_rooms=10, room_types=2, room_repetitions=2,
                 ade_path='../../data/ADE20K_2021_17_01/images/ADE/training/',
                 image_resolution=(360, 360),
                 captions="./localized_narratives/ade20k_train_captions.json"):
        # the dimensions of the map
        self.n = n
        self.m = m
        # the number of rooms on the map
        self.n_rooms = n_rooms
        # TODO add explanation for room types and repetitions (what do they mean)
        self.room_types = room_types
        self.room_repetitions = room_repetitions

        with open(captions, 'r') as f:
            self.dict_captions = json.load(f)

        #
        self.current_room = []
        self.current_room_name = ''

        # the question is the caption generated from a randomly sampled room
        self.question = ''
        self.target_room = ''

        # TODO represent observation space and action space as gym.spaces
        # TODO like in https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        self.total_available_actions = {0: 'north', 1: 'east', 2: 'south', 3: 'west', 4: 'select_room'}

        # the state consists of: current room as numpy ndarray of shape (, , 3),
        # target room question as string,
        # available actions as list of string, max len=5
        self.state = []

        # done is False as long as an episode is running
        # when an episode terminates room_found is turned to 1 when the correct was selected
        self.done = False
        self.room_found = 0

        self.ade_path = ade_path

        # keep track of the total steps taken and the return
        self.model_return = 0
        self.model_steps = 0
        # TODO add bool to track success of search

        self.image_resolution = image_resolution

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

        # generate question based on the sampled map
        self.question, self.target_room = self.generate_question_from_image(self.mw.target_room)

        self.current_room_name = path.relpath(initial_state[0], self.ade_path)
        self.current_room = self.load_image(initial_state[0], self.image_resolution)
        # directions contains the available actions in the current room as a string (returned as part of the state)
        # 'select_room is always added at the end
        # available_actions contains them as a list for internal use of the environment
        self.directions = initial_state[1] + f', {self.total_available_actions[4]}.'
        self.available_actions = self.directions[12:].split() # prepending fluff is removed, actions are seperated by ","

        # keep track of the total steps taken and the return
        self.model_return = 0
        self.model_steps = 0

        # return the initial state
        self.state = [self.current_room, self.question, self.directions]
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
        if self.current_room_name == self.target_room:
            reward = 1000.0
            self.room_found = 1
        else:
            reward = -1000.0
            self.room_found = 0
        # Terminate the game
        self.done = True
        self.state = [self.current_room, self.question, self.directions]

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
            reward = -1.0 * self.model_steps #-10.0 / (1 + np.exp(-self.model_steps + 5))
            state = self.mw.upd(action)
            self.current_room_name = path.relpath(state[0], self.ade_path)
            self.current_room = self.load_image(state[0], self.image_resolution)
            self.directions = state[1] + f', {self.total_available_actions[4]}.'
        else:
            # TODO not sure what the correct reward here would be for taking an unavailable action
            # TODO maybe stop penalizing wrong actions
            reward = 0.0

        self.state = [self.current_room, self.question, self.directions]

        return reward

    def generate_question_from_image(self, image_path):
        """
        Extracts a caption for an image to pose as the room to be found on the map
        :param image_path:
        :return: the captions and the name/category of the target room as a string
        """
        #
        target_room = path.relpath(image_path, self.ade_path)
        #
        caption_key = os.path.split(target_room)[1].strip('.jpg')
        target_caption = self.dict_captions[caption_key]
        return target_caption, target_room

    def load_image(self, image_path, image_resolution):
        """
        Loads an image from disk and reshapes it to a given resolution.
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
