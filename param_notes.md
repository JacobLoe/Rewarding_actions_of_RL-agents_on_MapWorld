# Parameters for MapWorld
```
    "ade_path": "../../data/ADE20K_2021_17_01/images/ADE/training/",
    "captions": "localized_narratives/ade20k_train_captions.json",
    "image_height": 360,
    "image_width": 360,
    "m": 4,
    "n": 4,
    "n_rooms": 10,
    "room_repetitions": 2,
    "room_types": 2,
```
These values set the properties of all maps.
```
    "reward_step": -10.0,
    "reward_linear_step": -0.6666,
    "reward_logistic_step": -10.0,
    "reward_wrong_action": 0.0,
    "reward_room_selection": 2000.0,
    "penalty_room_selection": -1000.0,
    "reward_selection_by_distance": "True",
    "reward_step_function": "linear"
```
`reward_step` and `reward_logistic_step` set the upper bound of the reward.
`reward_linear_step` defines the slope of the linear reward like `reward_linear_reward * number_of_steps`.
It is set up to reach -10.0 after 15 steps. 
The logistic reward function also reaches -10 after 15 steps and then stays at -10.

If `reward_selection_by_distance` is set to `True` the reward for selecting a room is only influenced by `reward_room_selection`.
In this case it defines the difference between the maximum and minimum reward. 

