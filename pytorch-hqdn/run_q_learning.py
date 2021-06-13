from envs.mdp import StochasticMDPEnv
from utils.plotting import plot_episode_stats, plot_visited_states
from utils.schedule import LinearSchedule

from q_learning import q_learning


if __name__ == '__main__':
    NUM_EPISODES = 12000
    exploration_schedule = LinearSchedule(50000, 0.1, 1.0)

    env = StochasticMDPEnv()

    Q, stats, visits = q_learning(env, NUM_EPISODES, lr=0.00025, exploration_schedule=exploration_schedule)

    # plot_episode_stats(stats)
    # plot_visited_states(visits, NUM_EPISODES)