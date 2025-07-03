import os
import tempfile
import xml.etree.ElementTree as ET

import gymnasium
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.envs.mujoco.reacher_v4 import ReacherEnv


class ReacherLengthEnv(ReacherEnv, utils.EzPickle):
    def __init__(
            self,
            body_parts=None,
            length=None,
            *args,
            **kwargs):

        if length is None:
            length = [0.1, 0.1]
        if body_parts is None:
            body_parts = ["link0", "link1"]
        assert isinstance(self, mujoco_env.MujocoEnv)

        super().__init__(**kwargs)
        model_path = os.path.join(os.path.dirname(gymnasium.envs.mujoco.__file__), 'assets', 'reacher.xml')
        # find the body_part we want
        tree = ET.parse(model_path)
        for body_part, leng in zip(body_parts, length):

            # grab the geoms
            geom = tree.find(".//geom[@name='%s']" % body_part)

            fromto = []
            for x in geom.attrib["fromto"].split(" "):
                fromto.append(float(x))
            fromto[3] = leng
            geom.attrib["fromto"] = " ".join([str(x) for x in fromto])

        # Modify the frame position of second link

        body_parts = ['body1', 'fingertip']
        lengths = [length[0], length[1]+0.01]

        for body_part, length in zip(body_parts, lengths):
            body = tree.find(".//body[@name='{}']".format(body_part))
            pos = []
            for x in body.attrib["pos"].split(" "):
                pos.append(float(x))
            pos[0] = length
            body.attrib['pos'] = " ".join([str(x) for x in pos])


        # create new xml
        _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)

        # load the modified xml
        mujoco_env.MujocoEnv.__init__(self, file_path, 2, None, render_mode=kwargs.get('render_mode', None))
        utils.EzPickle.__init__(self)

if __name__ == "__main__":
    env = ReacherLengthEnv(length=[0.5, 0.15], render_mode='human')
    env.reset()
    print(env.observation_space.sample())
    while True:
        env.render()
        env.step(env.action_space.sample())
