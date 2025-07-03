import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env


def GravityEnvFactory(class_type):
    """class_type should be an OpenAI gym type"""

    class GravityEnv(class_type, utils.EzPickle):
        """
        Allows the gravity to be changed by the
        """
        def __init__(
                self,
                model_path,
                gravity=-9.81,
                *args,
                **kwargs):
            class_type.__init__(self, model_path=model_path)
            utils.EzPickle.__init__(self)

            # make sure we're using a proper OpenAI gym Mujoco Env
            assert isinstance(self, mujoco_env.MujocoEnv)

            self.model.opt.gravity[:] = np.array([0., 0., gravity])

    return GravityEnv
