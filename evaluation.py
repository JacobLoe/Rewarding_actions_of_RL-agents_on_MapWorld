from agents import RandomBaseline
from MapWorld import MapWorldGym

if __name__ == '__main__':

    mwg = MapWorldGym()

    rb = RandomBaseline()

    for i in range(10):
        print(rb.select_action(5))