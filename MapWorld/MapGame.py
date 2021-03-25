from maps import ADEMap
from mapworld import MapWorld, MapWorldWrapper
import numpy as np
import gym
import cv2


# FIXME remove unnecessary 'outdoor' rooms from maps.py (church, street etc.)

# FIXME maybe move captioning, embeddings etc into agent

# FIXME make the returned actios of the env dynamic

# FIXME get correcte ADE20K datset from patrick/Brielen

class MapWorldGym(gym.Env):

    def __init__(self, n=4, m=4, n_rooms=10, room_types=2, room_repetitions=2):
        # the dimensions of the map
        self.n = n
        self.m = m
        # the number of rooms on the map
        self.n_rooms = n_rooms
        #
        self.room_types = room_types
        self.room_repetitions = room_repetitions

        # FIXME load im2txt model
        self.image_caption_model = 0

        #
        self.current_room = []
        self.current_room_source = ''   # FIXME find better name

        # the question is the caption generated from a randomly sampled room
        self.question = ''
        self.target_room = ''

        # FIXME those probably don't belong here when actions are dynamic
        # FIXME maybe only give the max num of possible actions
        self.available_actions = ['north', 'east', 'south', 'west', 'answer']
        self.num_actions = len(self.available_actions)

        # the state consists of: current room as numpy ndarray of shape (, , 3),
        # target room question as string,
        # available actions as list of string, max len=5
        self.state = []

        self.done = False

    def reset(self):
        """
        resets the environment to its initial values
        samples a new map, generates a question from the map
        :return: list, returns the current room, question and available actions
        """
        # initialise a ne MapWorld object with the parameters set in the init
        ade_map = ADEMap(self.n, self.m, self.n_rooms,
                         (self.room_types, self.room_repetitions))
        # FIXME add MapWorldWrapper
        self.mw = MapWorld(ade_map.to_fsa_def(), ['instance', 'type'])

        image_path = 'ADE20k_test.jpeg'

        # generate question based on the sampled map
        self.question = self.generate_question()
        self.target_room = image_path[:-5]  # FIXME get target room string from map

        # FIXME get randomised initial position instead of using test image
        self.current_room_source = image_path[:-5]
        self.current_room = np.array(cv2.imread(image_path))

        # FIXME get available actions from Map
        self.available_actions = ['north', 'east', 'south', 'west', 'answer']

        # return the initial state
        self.state = [self.current_room, self.question, self.available_actions]
        return self.state

    def step(self, action):
        """
        Take one step in the environement
        :param action: string, one of: north, east, south, west, answer
        :return: list, contains the state, reward and signal if the game is done
        """

        # FIXME penalize wrong actions
        if action == 'north':
            reward = -10.0
            # FIXME apply action to Map, get image and actions
            image_path = 'ADE20k_test.jpeg'
            self.current_room = np.array(cv2.imread(image_path))
            self.available_actions = ['north', 'east', 'south', 'west', 'answer']

            self.state = [self.current_room, self.question, self.available_actions]

        elif action == 'east':
            reward = -10.0
            # FIXME apply action to Map, get image and actions
            image_path = 'ADE20k_test.jpeg'
            self.current_room = np.array(cv2.imread(image_path))
            self.available_actions = ['north', 'east', 'south', 'west', 'answer']

            self.state = [self.current_room, self.question, self.available_actions]

        elif action == 'south':
            reward = -10.0
            # FIXME apply action to Map, get image and actions
            image_path = 'ADE20k_test.jpeg'
            self.current_room = np.array(cv2.imread(image_path))
            self.available_actions = ['north', 'east', 'south', 'west', 'answer']

            self.state = [self.current_room, self.question, self.available_actions]

        elif action == 'west':
            reward = -10.0
            # FIXME apply action to Map, get image and actions
            image_path = 'ADE20k_test.jpeg'
            self.current_room = np.array(cv2.imread(image_path))
            self.available_actions = ['north', 'east', 'south', 'west', 'answer']

            self.state = [self.current_room, self.question, self.available_actions]

        elif action == 'answer':

            if self.current_room_source == self.target_room:
                reward = 100.0
            else:
                reward = -100.0
            # Terminate the game
            self.done = True

        self.state = [self.current_room, self.question, self.available_actions]

        return [self.state, reward, self.done, {}]   # no clue what the point of the dict is

    def generate_question(self):
        """
        Generate a question string from a image
        """
        # FIXME sample random image from self.mw instead of test image
        self.target_room = 'ADE20k_test'    # save target room for later comparison
        sample_image_path = '{}.jpeg'.format(self.target_room)
        sample_room = cv2.imread(sample_image_path)

        question = 'self.image_caption_model(sample_room)'
        return question

    def render(self, mode='human'):
        # FIXME probably not really necessary
        pass


if __name__ == '__main__':

    m = MapWorldGym()
    i = m.reset()
    for a in ['north', 'east', 'south', 'west', 'answer']:
        print(m.step(a))
