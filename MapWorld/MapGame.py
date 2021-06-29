from MapWorld.maps import ADEMap
from MapWorld.mapworld import MapWorldWrapper
from im2txt import Captioning
import numpy as np
import gym
import cv2
from os import path


# TODO maybe accelerate game by loading all images into cache before hand


class MapWorldGym(gym.Env):

    def __init__(self, n=4, m=4, n_rooms=10, room_types=2, room_repetitions=2,
                 ade_path='../ADE20K_2021_17_01/images/ADE/training/',
                 caption_checkpoints="./im2txt/checkpoints/im2txt_5m/model.ckpt-5000000",
                 caption_vocab='./im2txt/vocab/word_counts.txt',
                 image_resolution=(480, 854)):
        # TODO possibly kick out useless variables from init
        # the dimensions of the map
        self.n = n
        self.m = m
        # the number of rooms on the map
        self.n_rooms = n_rooms
        # TODO add explanation for room types and repetitions (what do they mean)
        self.room_types = room_types
        self.room_repetitions = room_repetitions

        self.image_caption_model = Captioning(caption_checkpoints,
                                              caption_vocab)

        #
        self.current_room = []
        self.current_room_name = ''

        # the question is the caption generated from a randomly sampled room
        self.question = ''
        self.target_room = ''

        # TODO represent observation space and action space as gym.spaces
        # TODO like in https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        self.total_available_actions = ['north', 'east', 'south', 'west', 'answer']
        self.total_num_actions = len(self.total_available_actions)

        # the state consists of: current room as numpy ndarray of shape (, , 3),
        # target room question as string,
        # available actions as list of string, max len=5
        self.state = []

        self.done = False

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

        # TODO rescale images to consistent resolution
        self.current_room_name = path.relpath(initial_state[0], self.ade_path)
        self.current_room = self.load_image(initial_state[0], self.image_resolution)

        self.directions = initial_state[1] + ', answer.'
        self.available_actions = self.directions[12:].split()

        # keep track of the total steps taken and the return
        self.model_return = 0
        self.model_steps = 0

        # TODO check whether the outputs are numpy arrays (or strings)
        # return the initial state
        self.state = [self.current_room, self.question, self.directions]
        return self.state

    def step(self, action):
        """
        Take one step in the environment
        :param action: string, one of: north, east, south, west, answer
        :return: list, contains the state, reward and signal if the game is done
        """

        if action == 'answer':
            if self.current_room_name == self.target_room:
                reward = 100.0
            else:
                reward = -100.0
            # Terminate the game
            self.done = True
            self.state = [self.current_room, self.question, self.directions]
        else:
            reward = self.move(action)

        self.model_return += reward
        self.model_steps += 1

        return self.state, reward, self.done, {}   # dict is used to convey info

    def move(self, action):
        """
        Moves in the map according to the action. If the action is not in the available actions stays in the current
        room and penalizes heavily.
        Args:
            action: string, one from north, south, west, east

        Returns: a float, the reward for the action
        """
        if action in self.available_actions:
            reward = -10.0
            state = self.mw.upd(action)
            self.current_room_name = path.relpath(state[0], self.ade_path)
            self.current_room = np.array(cv2.imread(state[0]))
            self.directions = state[1] + ', answer'
        else:
            reward = -100.0

        self.state = [self.current_room, self.question, self.directions]

        return reward

    def generate_question_from_image(self, image_path):
        """
        Extracts a caption for an image to pose as the room to be found on the map
        :param image_path:
        :return: the captions and the name/category of the target room as a string
        """
        target_room = path.relpath(image_path, self.ade_path)
        question = self.image_caption_model.image(image_path)['1']['Sentence']
        # capitalize first letter, remove trailing space and stop, add stop at proper place
        question = question.capitalize().strip('.').strip() + '.'
        return question, target_room

    def load_image(self, image_path, image_resolution):
        """
        Loads an image from disk and reshapes while keeping the aspect ration intact.
        Args:
            image_path: strind
            image_width: integer

        Returns: Numpy array, reshaped image, height, width, channels,
        """
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_resolution)
        image = np.array(image)

        return image

    def render(self, mode='human'):
        # FIXME probably not really necessary
        pass
