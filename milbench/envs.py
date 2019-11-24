"""Gym wrapper for shape-pushing environment."""

import math

import gym
from gym import spaces
from gym.wrappers import FrameStack, ResizeObservation
import numpy as np
import pymunk as pm

import milbench.entities as en
import milbench.gym_render as r


class ShapePushingEnv(gym.Env):
    def __init__(self, res_hw=(256, 256), fps=20, phys_steps=10, phys_iter=10):
        self.phys_iter = phys_iter
        self.phys_steps = phys_steps
        self.fps = fps
        self.res_hw = res_hw
        # RGB observation, stored as bytes
        self.observation_space = spaces.Box(low=0.0,
                                            high=255,
                                            shape=(*res_hw, 3),
                                            dtype=np.uint8)
        # First axis is movement of [none, up, down], second axis is rotation
        # of [none, left, right], third axis is gripper state of [open, close].
        self.action_space = spaces.MultiDiscrete([3, 3, 2])
        self._ac_flags_ud = [
            en.RobotAction.NONE, en.RobotAction.UP, en.RobotAction.DOWN
        ]
        self._ac_flags_lr = [
            en.RobotAction.NONE, en.RobotAction.LEFT, en.RobotAction.RIGHT
        ]
        self._ac_flags_grip = [en.RobotAction.OPEN, en.RobotAction.CLOSE]

        # state/rendering (see reset())
        self._entities = None
        self._space = None
        self._robot = None
        # this is for background rendering
        self.viewer = None

    # def seed(self, seed=None):
    #     """No randomness for now (except in action spaces, which must be
    #     seeded separately)."""
    #     return []

    def reset(self):
        if self._entities is not None:
            # delete old entities/space
            self._entities = []
            self._space = None
            self._robot = None
        if self.viewer is None:
            res_h, res_w = self.res_hw
            self.viewer = r.Viewer(res_w, res_h, visible=False)
        else:
            # these will get added back later
            self.viewer.reset_geoms()
        self._space = pm.Space()
        self._space.iterations = self.phys_iter

        # set up robot and arena
        robot_rad = 0.2
        shape_rad = robot_rad * 2 / 3
        self._robot = robot = en.Robot(radius=robot_rad,
                                       init_pos=(0.1, -0.1),
                                       init_angle=math.pi / 9,
                                       mass=1.0)
        arena = en.ArenaBoundaries(left=-1.0, right=1.0, bottom=-1.0, top=1.0)
        square = en.Shape(shape_type=en.ShapeType.SQUARE,
                          colour_name='red',
                          shape_size=shape_rad,
                          init_pos=(0.4, -0.4),
                          init_angle=0.13 * math.pi)
        # square2 = en.Shape(shape_type=en.ShapeType.SQUARE,
        #                    colour_name='red',
        #                    shape_size=shape_rad,
        #                    init_pos=(0.25, -0.65),
        #                    init_angle=0.23 * math.pi)
        # circle = en.Shape(shape_type=en.ShapeType.CIRCLE,
        #                   colour_name='yellow',
        #                   shape_size=shape_rad,
        #                   init_pos=(-0.7, -0.5),
        #                   init_angle=-0.5 * math.pi)
        # triangle = en.Shape(shape_type=en.ShapeType.HEXAGON,
        #                     colour_name='green',
        #                     shape_size=shape_rad,
        #                     init_pos=(-0.5, 0.35),
        #                     init_angle=0.05 * math.pi)
        # pentagon = en.Shape(shape_type=en.ShapeType.PENTAGON,
        #                     colour_name='blue',
        #                     shape_size=shape_rad,
        #                     init_pos=(0.4, 0.35),
        #                     init_angle=0.8 * math.pi)
        # self._entities = [circle, square, triangle, pentagon, robot, arena]
        self._entities = [square, robot, arena]

        for ent in self._entities:
            ent.setup(self.viewer, self._space)

        self.viewer.set_bounds(left=arena.left,
                               right=arena.right,
                               bottom=arena.bottom,
                               top=arena.top)

        # # step forward by one second so PyMunk can recover from bad initial
        # # conditions
        # forward_time = 1.0
        # forward_frames = int(math.ceil(forward_time * self.fps))
        # for _ in range(forward_frames):
        #     self._phys_steps_on_frame()

        return self.render(mode='rgb_array')

    def _phys_steps_on_frame(self):
        phys_steps = 10
        spf = 1 / self.fps
        dt = spf / phys_steps
        for i in range(phys_steps):
            for ent in self._entities:
                ent.update(dt)
            self._space.step(dt)

    def step(self, action):
        # step forward physics
        ac_ud, ac_lr, ac_grip = action
        action_flag = en.RobotAction.NONE
        action_flag |= self._ac_flags_ud[ac_ud]
        action_flag |= self._ac_flags_lr[ac_lr]
        action_flag |= self._ac_flags_grip[ac_grip]
        self._robot.set_action(action_flag)
        self._phys_steps_on_frame()
        reward = 0.0
        done = False
        info = {}
        obs_u8 = self.render(mode='rgb_array')
        return obs_u8, reward, done, info

    def render(self, mode='human'):
        if self.viewer is None:
            return None
        for ent in self._entities:
            ent.pre_draw()
        if mode == 'human':
            self.viewer.window.set_visible(True)
        else:
            assert mode == 'rgb_array'
        return self.viewer.render(return_rgb_array=True)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def register():
    default_res = (256, 256)
    small_res = (96, 96)
    default_kwargs = dict(res_hw=default_res,
                          fps=15,
                          phys_steps=10,
                          phys_iter=10)
    # 250 frames is 20s at 15fps
    ep_len = 200
    gym.register('ShapePush-v0',
                 entry_point='milbench.envs:ShapePushingEnv',
                 max_episode_steps=ep_len,
                 kwargs=default_kwargs)

    def make_lores_stack(**kwargs):
        base_env = ShapePushingEnv(**kwargs)
        resize_env = ResizeObservation(base_env, small_res)
        stack_env = FrameStack(resize_env, 4)
        return stack_env

    # images downsampled to 128x128, four adjacent frames stacked together
    gym.register('ShapePushLoResStack-v0',
                 entry_point=make_lores_stack,
                 max_episode_steps=ep_len,
                 kwargs=default_kwargs)
