import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Trajectory-v0',
    entry_point='gym_trajectory.envs:TrajectoryEnv',
    # reward_threshold=1.0,
    nondeterministic=False
)