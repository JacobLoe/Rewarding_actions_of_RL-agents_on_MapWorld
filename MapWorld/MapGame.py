from MapWorld.maps import ADEMap
from MapWorld.mapworld import MapWorldWrapper
from im2txt import Captioning
import numpy as np
import gym
import cv2
from os import path


# TODO make the returned actions of the env dynamic

# TODO maybe accelerate game by loading all images into cache before hand


class MapWorldGym(gym.Env):

    def __init__(self, n=4, m=4, n_rooms=10, room_types=2, room_repetitions=2,
                 ade_path='../ADE20K_2021_17_01/images/ADE/training/',
                 caption_checkpoints="./im2txt/checkpoints/im2txt_5m/model.ckpt-5000000",
                 caption_vocab='./im2txt/vocab/word_counts.txt'):
        # TODO possibly kick out useless variables from init
        # the dimensions of the map
        self.n = n
        self.m = m
        # the number of rooms on the map
        self.n_rooms = n_rooms
        #
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

    def reset(self):
        """
        resets the environment to its initial values
        samples a new map, generates a question from the map
        :return: list, returns the current room, question and available actions
        """
        self.done = False
        # initialise a ne MapWorld object with the parameters set in the init
        ade_map = ADEMap(self.n, self.m, self.n_rooms, (self.room_types, self.room_repetitions))
        self.mw = MapWorldWrapper(ade_map, image_prefix=self.ade_path)
        initial_state = self.mw.initial_state

        # generate question based on the sampled map
        self.question, self.target_room = self.generate_question_from_image(self.mw.target_room)

        # TODO rescale images to consistent resolution
        self.current_room_name = path.relpath(initial_state[0], self.ade_path)
        self.current_room = np.array(cv2.imread(initial_state[0]))

        self.available_actions = initial_state[1] + ['answer']

        # keep track of the total steps taken and the return
        self.model_return = 0
        self.model_steps = 0

        # return the initial state
        self.state = [np.shape(self.current_room), self.question, self.available_actions]
        return self.state

    def step(self, action):
        """
        Take one step in the environment
        :param action: string, one of: north, east, south, west, answer
        :return: list, contains the state, reward and signal if the game is done
        """

        # FIXME penalize wrong actions
        if action == 'north':
            reward = -10.0
            state = self.mw.upd(action)
            self.current_room_name = path.relpath(state[0], self.ade_path)
            self.current_room = np.array(cv2.imread(state[0]))
            self.available_actions = state[1] + ['answer']

            self.state = [np.shape(self.current_room), self.question, self.available_actions]

        elif action == 'east':
            reward = -10.0
            state = self.mw.upd(action)
            self.current_room_name = path.relpath(state[0], self.ade_path)
            self.current_room = np.array(cv2.imread(state[0]))
            self.available_actions = state[1] + ['answer']

            self.state = [np.shape(self.current_room), self.question, self.available_actions]

        elif action == 'south':
            reward = -10.0
            state = self.mw.upd(action)
            self.current_room_name = path.relpath(state[0], self.ade_path)
            self.current_room = np.array(cv2.imread(state[0]))
            self.available_actions = state[1] + ['answer']

            self.state = [np.shape(self.current_room), self.question, self.available_actions]

        elif action == 'west':
            reward = -10.0
            state = self.mw.upd(action)
            self.current_room_name = path.relpath(state[0], self.ade_path)
            self.current_room = np.array(cv2.imread(state[0]))
            self.available_actions = state[1] + ['answer']

            self.state = [np.shape(self.current_room), self.question, self.available_actions]

        elif action == 'answer':

            if self.current_room_name == self.target_room:
                reward = 100.0
            else:
                reward = -100.0
            # Terminate the game
            self.done = True
            self.state = [self.current_room, self.question, self.available_actions]

        self.model_return += reward
        self.model_steps += 1

        return [self.state, reward, self.done, {}]   # dict is used to convey info

    def generate_question_from_image(self, image_path):
        """
        Extracts a caption for an image to pose as the room to be found on the map
        :param image_path:
        Generate a question string from an image
        :return: the captions and the name/category of the target room as a string
        """
        target_room = path.relpath(image_path, self.ade_path)
        question = self.image_caption_model.image(image_path)['1']['Sentence']
        # print('image_path', image_path)
        # print('question', question)
        return question, target_room

    def render(self, mode='human'):
        # FIXME probably not really necessary
        pass
