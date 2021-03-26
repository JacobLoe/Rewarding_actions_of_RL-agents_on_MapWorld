from agents import RandomBaseline
from MapWorld import MapWorldGym

if __name__ == '__main__':

    mwg = MapWorldGym()
    mwg.reset()

    available_actions = mwg.available_actions
    num_actions = mwg.num_actions
    print(num_actions, available_actions)
    rb = RandomBaseline()

    while not mwg.done:
        action = available_actions[rb.select_action(5)]
        s = mwg.step(action)
        print(s[1:])

