from maps import AbstractMap, ADEMap
from mapworld import MapWorld, MapWorldWrapper
import numpy as np
import gym
import cv2


# FIXME remove unnecessary 'outdoor' rooms from map (church, street etc.)

class MapWorldGym(gym.Env):

    def __init__(self):
        # the dimensions of the map
        self.n = 4
        self.m = 4
        # the number of rooms on the map
        self.n_rooms = 10
        #
        self.room_types = 2
        self.room_repetitions = 2

        self.done = False

        #
        self.object_detection_model = 0
        self.image_caption_model = 0

        self.question = ''

    def reset(self):
        """
        resets the environment to its initial values
        """
        # initialise a ne MapWorld object with the parameters set in the init
        ade_map = ADEMap(self.n, self.m, self.n_rooms,
                         (self.room_types, self.room_repetitions))
        # FIXME add MapWorldWrapper
        self.mw = MapWorld(ade_map.to_fsa_def(), ['instance', 'type'])

        self.generate_question()

        state = self.get_state(image)

        available_actions = []

        # return the initial observation
        return [state, self.question, available_actions]

    def step(self, action):

        if action == 'n':
            reward = -10.0
            state = self.mw.try_transition(action)

        elif action == 'o':
            reward = -10.0
            state = self.mw.try_transition(action)

        elif action == 's':
            reward = -10.0
            state = self.mw.try_transition(action)

        elif action == 'w':
            reward = -10.0
            state = self.mw.try_transition(action)

        elif action == 'answer':

            reward = 100.0
            self.done = True
            reward = -100.0
            self.done = True

        return [state, reward, self.done, {}] # add

    def get_captions(self, image):

        captions = self.image_caption_model(image)

        return captions

    def get_state(self, image):

        state = ''

        return state

    def answer_question(self):

        done = False

        return done

    def generate_question(self):
        # sample random image from self.mw
        # FIXME

        image = cv2.imread()

        question = self.get_captions(image)

        self.question = question

    def render(self, mode='human'):
        # FIXME probably not really necessary
        pass


if __name__ == '__main__':
    ade_map = ADEMap(4, 4, 10, (2, 2))

    mw = MapWorld(ade_map.to_fsa_def(), ['instance', 'type'])

    state_description = mw.try_transition('n')

    print(state_description)