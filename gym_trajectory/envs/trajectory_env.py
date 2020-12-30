import numpy as np
import gym
from gym import utils
import gym.spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering



class TrajectoryEnv(gym.Env, utils.EzPickle):
    """Custom environment for OpenAI gym
    """
    metadata = {'render.modes': ['ansi', 'rgb_array', 'human']}

    VIEWER_WIDTH = 600
    VIEWER_HEIGHT = 400

    AGENT_COLOR = (1, 0, 0)
    AGENT_SIZE = 5

    TARGET_COLOR = (0, 1, 0)
    TARGET_SIZE = 5

    OBSERVABLE_COLOR = (0, 0, 1)
    OBSERVABLE_SIZE = 3


    def __init__(
        self,
        num_dimensions = 2,
        num_observables = 5,
        max_targets = 100,
        max_position = 100.0,
        max_acceleration = 10.1,
        max_velocity = 2.0,
        collision_epsilon = 25.0
        ) -> None:
        super().__init__()

        self.num_dimensions = num_dimensions
        self.num_observables = num_observables
        
        self.action_space = gym.spaces.Box(
            low = -max_acceleration,
            high = max_acceleration,
            shape = (self.num_dimensions,)
        )

        self.position_space = gym.spaces.Box(
            low = -max_position,
            high = max_position,
            shape = (self.num_dimensions,)
        )

        self.velocity_space = gym.spaces.Box(
            low = -max_velocity,
            high = max_velocity,
            shape = (self.num_dimensions,)
        )

        self.target_space = gym.spaces.Tuple(
            [self.position_space for _ in range(self.num_observables)]
        )

        self.observation_space = gym.spaces.Tuple(
            (self.position_space, self.velocity_space, self.target_space)
        )

        self.max_targets = max_targets
        self.num_targets = 0
        self.previous_distance = 0.0
        self.agent_position = None
        self.agent_velocity = None
        self.target_position = None
        self.done = False
        self.collision_epsilon = collision_epsilon


        self.viewer = None
        self.viewer_offset = (max_position, max_position)
        self.viewer_scale = (self.VIEWER_WIDTH / (2*max_position), self.VIEWER_HEIGHT / (2*max_position))

        self.viewer_agent = None
        self.viewer_agent_transform = None

        self.viewer_target = None
        self.viewer_target_transform = None

        self.viewer_observable = []
        self.viewer_observable_transform = []



    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        info = {}
        reward = 0.0

        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.agent_velocity = np.clip(
            self.agent_velocity * action,
            self.velocity_space.low, self.velocity_space.high)

        self.agent_position = np.clip(
            self.agent_position + self.agent_velocity,
            self.position_space.low, self.position_space.high)

        target_distance = np.linalg.norm(self.agent_position - self.target_position[0])
        max_distance = np.linalg.norm(self.position_space.high - self.position_space.low)

        if target_distance < self.previous_distance:
            reward = max_distance/(target_distance + 1)
            self.previous_distance = target_distance
        else:
            reward = -max_distance/(target_distance + 1)

        if target_distance < self.collision_epsilon:
            reward *= 2
            self.target_position = self.target_position[1:] + (self.position_space.sample(),)
            self.previous_distance = np.linalg.norm(self.agent_position - self.target_position[0])
            self.num_targets += 1
            if self.num_targets >= self.max_targets:
                self.done = True

        reward = np.clip(reward, self.reward_range[0], self.reward_range[1])
        observation = (self.agent_position, self.agent_velocity, self.target_position)
        return (observation, reward, self.done, info)



    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """

        self.done = False
        self.num_targets = 0
        self.agent_position, self.agent_velocity, self.target_position = self.observation_space.sample()
        self.previous_distance = np.linalg.norm(self.agent_position - self.target_position[0])
        return (self.agent_position, self.agent_velocity, self.target_position)



    def render(self, mode='human'):
        """Renders the environment.

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Args:
            mode (str): the mode to render with
        """
        if mode == 'ansi':
            print("distance: {}".format(np.linalg.norm(self.agent_position - self.target_position[0])))
        elif (mode == 'human' or mode == 'rgb_array') and self.num_dimensions == 2:
            
            if self.viewer is None:
            
                # viewer:
                self.viewer = rendering.Viewer(self.VIEWER_WIDTH, self.VIEWER_HEIGHT)
                
                # agent:
                self.viewer_agent_transform = rendering.Transform()
                self.viewer_agent = rendering.make_circle(self.AGENT_SIZE)
                self.viewer_agent.set_color(*self.AGENT_COLOR)
                self.viewer_agent.add_attr(self.viewer_agent_transform)
                self.viewer.add_geom(self.viewer_agent)

                # target:
                self.viewer_target_transform = rendering.Transform()
                self.viewer_target = rendering.make_circle(self.TARGET_SIZE)
                self.viewer_target.set_color(*self.TARGET_COLOR)
                self.viewer_target.add_attr(self.viewer_target_transform)
                self.viewer.add_geom(self.viewer_target)

                # other observable targets:
                for i in range(self.num_observables - 1):
                    observable_transform = rendering.Transform()
                    observable = rendering.make_circle(self.OBSERVABLE_SIZE)
                    observable.set_color(*self.OBSERVABLE_COLOR)
                    observable.add_attr(observable_transform)
                    self.viewer.add_geom(observable)
                    self.viewer_observable.append(observable)
                    self.viewer_observable_transform.append(observable_transform)

            self.viewer_agent_transform.set_translation(
                *(self.viewer_scale * (self.agent_position + self.viewer_offset)))
            self.viewer_target_transform.set_translation(
                *(self.viewer_scale * (self.target_position[0] + self.viewer_offset)))
            
            line = rendering.Line(
                start=(self.viewer_scale * (self.agent_position + self.viewer_offset)),
                end=(self.viewer_scale * (self.target_position[0] + self.viewer_offset)))
            self.viewer.add_onetime(line)
            for idx, observable_transform in enumerate(self.viewer_observable_transform):
                self.viewer_observable[idx].set_color(
                    0,
                    1 - idx * 1/len(self.viewer_observable_transform),
                    idx * 1/len(self.viewer_observable_transform))
                observable_transform.set_translation(
                    *(self.viewer_scale * (self.target_position[idx+1] + self.viewer_offset)))

                line = rendering.Line(
                    start=(self.viewer_scale * (self.target_position[idx] + self.viewer_offset)),
                    end=(self.viewer_scale * (self.target_position[idx+1] + self.viewer_offset)))
                self.viewer.add_onetime(line)

            return self.viewer.render(return_rgb_array = mode == 'rgb_array')

        else:
            super(TrajectoryEnv, self).render(mode=mode)