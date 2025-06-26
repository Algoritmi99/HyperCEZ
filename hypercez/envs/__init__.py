import os

from gymnasium.envs.registration import register


register(
    id='MBRLCartpole-v0',
    entry_point='hypercez.envs.cartpole:CartpoleEnv',
    max_episode_steps=200
)

register(
    id='MBRLHalfCheetah-v0',
    entry_point='hypercez.envs.half_cheetah:HalfCheetahEnv',
    max_episode_steps=1000
)

register(
    id='MBRLHopper-v0',
    entry_point='hypercez.envs.hopper:HopperEnv',
    reward_threshold=None,
    max_episode_steps=1000
)

register(
    id='MBRLWalker-v0',
    entry_point='hypercez.envs.walker2d:Walker2dEnv',
    reward_threshold=None,
    max_episode_steps=1000
)

register(
    id='CartpoleLeft1-v0',
    entry_point='hypercez.envs.cartpole:CartpoleBinEnv',
    reward_threshold=None,
    max_episode_steps=200,
    kwargs=dict(offset=-5)
)

register(
    id='CartpoleRight1-v0',
    entry_point='hypercez.envs.cartpole:CartpoleBinEnv',
    reward_threshold=None,
    max_episode_steps=200,
    kwargs=dict(offset=5)
)

register(
    id='CartpoleLong1-v0',
    entry_point='hypercez.envs.cartpole:CartpoleLengthEnv',
    reward_threshold=None,
    max_episode_steps=200,
    kwargs=dict(length=0.8)
)

register(
    id='CartpoleLong2-v0',
    entry_point='hypercez.envs.cartpole:CartpoleLengthEnv',
    reward_threshold=None,
    max_episode_steps=200,
    kwargs=dict(length=1.0)
)

register(
    id='CartpoleLong3-v0',
    entry_point='hypercez.envs.cartpole:CartpoleLengthEnv',
    reward_threshold=None,
    max_episode_steps=200,
    kwargs=dict(length=1.2)
)

register(
    id='CartpoleLong4-v0',
    entry_point='hypercez.envs.cartpole:CartpoleLengthEnv',
    reward_threshold=None,
    max_episode_steps=200,
    kwargs=dict(length=0.7)
)

register(
    id='CartpoleLong5-v0',
    entry_point='hypercez.envs.cartpole:CartpoleLengthEnv',
    reward_threshold=None,
    max_episode_steps=200,
    kwargs=dict(length=0.9)
)

register(
    id='CartpoleLong6-v0',
    entry_point='hypercez.envs.cartpole:CartpoleLengthEnv',
    reward_threshold=None,
    max_episode_steps=200,
    kwargs=dict(length=1.1)
)

register(
    id='CartpoleLong7-v0',
    entry_point='hypercez.envs.cartpole:CartpoleLengthEnv',
    reward_threshold=None,
    max_episode_steps=200,
    kwargs=dict(length=1.3)
)

register(
    id='CartpoleShort1-v0',
    entry_point='hypercez.envs.cartpole:CartpoleLengthEnv',
    reward_threshold=None,
    max_episode_steps=200,
    kwargs=dict(length=0.4)
)

register(
    id='CartpoleShort2-v0',
    entry_point='hypercez.envs.cartpole:CartpoleLengthEnv',
    reward_threshold=None,
    max_episode_steps=200,
    kwargs=dict(length=0.5)
)

register(
    id='DoorLever-v0',
    entry_point='hypercez.envs.doorenv:DoorEnv',
    max_episode_steps=500,
    kwargs=dict(xml=os.path.dirname(os.path.realpath(__file__)) + '/assets/door/1551848929_lever_blue_right_v2_gripper_position.xml')
)

register(
    id='DoorPull-v0',
    entry_point='hypercez.envs.doorenv:DoorEnv',
    max_episode_steps=500,
    kwargs=dict(xml=os.path.dirname(os.path.realpath(__file__)) + '/assets/door/1555111990_pull_blue_right_v2_gripper_position.xml')
)

register(
    id='DoorRound-v0',
    entry_point='hypercez.envs.doorenv:DoorEnv',
    max_episode_steps=500,
    kwargs=dict(xml=os.path.dirname(os.path.realpath(__file__)) + '/assets/door/1558288438_round_blue_right_v2_gripper_position.xml')
)