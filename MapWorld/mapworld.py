# -*- coding: utf-8 -*-

'''The environment controller for MapWorld.
'''

from transitions import Machine
from numpy import random
from os import path


class MapWorld(object):
    '''The MapWorld environment. State machine for one agent.

    Tries to be general in what it returns when entering a new state.
    This is specified by the list of node_descriptors, which are fields
    in the dictionaries describing the nodes / states / rooms.
    '''
    def __init__(self, map_, node_descriptors):
        self.machine = Machine(model=self,
                               states=[str(this_node['id']) for this_node in map_['nodes']],
                               transitions=map_['transitions'],
                               ignore_invalid_triggers=True,
                               initial=map_['initial'])
        self.nodes = map_['nodes']
        self.node_descriptors = node_descriptors

    def _get_node(self, id_):
        for this_node in self.nodes:
            if str(this_node['id']) == id_:
                return this_node

    def describe_node(self, state):
        out = {}
        for descriptor in self.node_descriptors:
                out[descriptor] = (self._get_node(state)[descriptor])
        return (out, [t for t in self.machine.get_triggers(state)
                      if t in 'north south east west'.split()])

    def try_transition(self, trigger):
        if trigger not in self.machine.get_triggers(self.state):
            return (None,
                    [t for t in self.machine.get_triggers(self.state)
                     if t in 'north south east west'.split()])
        else:
            self.trigger(trigger)
            return self.describe_node(self.state)


class MapWorldWrapper(object):
    def __init__(self, map_, node_descriptor=['instance'], image_prefix=None):
        self.map = map_
        self.mw = MapWorld(map_.to_fsa_def(), node_descriptor)
        self.image_prefix = image_prefix
        # need to describe the initial state as well
        self.initial_state = self.describe_state(self.mw.state)
        # get a random room as the target
        self.target_room = self.get_target_room(self.mw.nodes)

    def describe_state(self, state):
        """

        :param state:
        Returns:

        """
        description, avail_dirs = self.mw.describe_node(state)
        image_path = path.join(self.image_prefix, description['instance'])
        # TODO don't return a list
        return [image_path, self.print_dirs(avail_dirs)]

    def print_dirs(self, avail_dirs):
        # TODO use comma as separator instead of spaces ?
        out_string = 'You can go: {}'.format(' '.join(avail_dirs))
        return out_string

    def upd(self, command):
        # TODO rename function,
        """

        Args:
            command:

        Returns:

        """
        if command == 'l':  # look: repeat directions, but don't show image again
            self.describe_state(self.mw.state)
        elif command in 'north south east west'.split():
            _, _ = self.mw.try_transition(command)
            state = self.describe_state(self.mw.state)
        # TODO don't return list
        return state

    def get_target_room(self, nodes):
        """
        Takes a random room from the map and returns its path
        :param nodes: list of dictionaries, has to at least contain the id (coordinate on the map)
        Returns: string, the path to the room
        """
        room_id = str(random.choice(nodes)['id'])
        return self.describe_state(room_id)[0]

    def plt(self):
        self.map.plot_graph(state=eval(self.mw.state))
