import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from math import hypot

class ThrowerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self._ball_hit_ground = False
        self._ball_hit_location = None
        mujoco_env.MujocoEnv.__init__(self, 'thrower.xml', 1)

    def _step(self, a):
        if not isinstance(a,int):
            a = 0
        ball_xy = self.get_body_com("ball")[:2]
        goal_xy = self.get_body_com("goal")[:2]
        reward = -100
        if not self._ball_hit_ground and self.get_body_com("ball")[2] < -0.25:
            self._ball_hit_ground = True
            self._ball_hit_location = self.get_body_com("ball")

        if self._ball_hit_ground:
            ball_hit_xy = self._ball_hit_location[:2]
            reward_dist  = -np.linalg.norm(ball_hit_xy - goal_xy)
            reward = - hypot(ball_hit_xy[0]-goal_xy[0], ball_hit_xy[1]-goal_xy[1])
        else:
            reward_dist = 0
            #reward_dist = -np.linalg.norm(ball_xy - goal_xy)
        reward_ctrl = - np.square(a).sum()
        actions = [
            [3.0,-20,1.5,-18.0],
            [-2.0,-20,1.5,-18.0],
            [3.0,-17,1.5,-18.0],
            [-2.0,-17,1.5,-18.0],
            [3.0,-17,1.5,-18.0],
            [3.0,-20,1.5,-10.0],
            [-2.0,-20,1.5,-10.0],
            [3.0,-17,1.5,-10.0],
            [-2.0,-17,1.5,-10.0],
            [-3.0,-17,1.5,-10.0],
            [0, 0,1,0],
            [0, 0,3,0]]
        if reward > - 0.1:
            print ("BUCKETS")
            reward = 10
        #print (a)
        self.do_simulation(actions[a], self.frame_skip)

        ob = self._get_obs()
        done = False
        if reward > 0.5:
            done = True
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        self._ball_hit_ground = False
        self._ball_hit_location = None

        qpos = self.init_qpos
        self.goal = [0.465, -0.328]
        #np.array( self.np_random.uniform(low=-0.3, high=0.3),self.np_random.uniform(low=-0.3, high=0.3)])

        qpos[-9:-7] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[7:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            #self.model.data.qpos.flat[:7],
            #self.model.data.qvel.flat[:7],
            self.get_body_com("r_wrist_roll_link"),
            self.get_body_com("ball"),
            #self.get_body_com("goal"),
        ])
