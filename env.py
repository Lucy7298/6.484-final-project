from pybullet_envs.gym_locomotion_envs import HopperBulletEnv
from model import LearnHopperPenalty
import numpy as np 
import torch
from collections import namedtuple

class Hopper(HopperBulletEnv):
    electricity_cost = -0.001  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
    joints_at_limit_cost = -0.1  # discourage stuck joints
    strain_cost = -0.0001
    electricity_surprise_weight = 1
    strain_surprise_weight = 1
    lambda1_prime = 2
    lambda2_prime = 2

    lambda1 = 2
    lambda2 = 2 
  
    def __init__(self, 
                 use_progress_reward, 
                 use_electricity_cost, 
                 use_limits_cost, 
                 use_strain_cost, 
                 use_electricity_surprise, 
                 use_strain_surprise, 
                 render=False, 
                 episode_steps=1000):
        """Modifies `__init__` in `HopperBulletEnv` parent class."""
        self.episode_steps = episode_steps
        self.use_progress_reward = use_progress_reward
        self.use_electricity_cost = use_electricity_cost
        self.use_limit_cost = use_limits_cost
        self.use_strain_cost = use_strain_cost
        self.use_electricity_surprise = use_electricity_surprise
        self.use_strain_surprise = use_strain_surprise
        self.prev_state_memory = {}

        super().__init__(render=render)

        self.torque_enabled = False

    def reset(self):
        """Modifies `reset` in `WalkerBaseBulletEnv` base class."""
        self.step_counter = 0
        self.ensemble_training_datas = []
        self.penalty_ensembles = []

        if self.use_electricity_surprise:
            self.ensemble_training_datas.append([])
            self.penalty_ensembles.append([LearnHopperPenalty(seed=idx) for idx in range(5)]) #DeepNormal models go here)
        if self.use_strain_surprise:
            self.ensemble_training_datas.append([])
            self.penalty_ensembles.append([LearnHopperPenalty(seed=idx) for idx in range(5, 10)]) #DeepNormal models go here)
    
        ret_val = super().reset()

        enable_torque = self.use_strain_cost or self.use_strain_surprise
        if enable_torque and not self.torque_enabled: 
            for joint in self.ordered_joints: 
                joint._p.enableJointForceTorqueSensor(joint.bodies[joint.bodyIndex], 
                                                  joint.jointIndex, 
                                                  1)
            self.torque_enabled = True

        # enable torque sensors 
        return ret_val

    def _isDone(self):
        """Modifies `_isDone` in `WalkerBaseBulletEnv` base class."""
        return (self.step_counter == self.episode_steps
            or super()._isDone())

    def compute_reward(self, old_state, a, state): 
        total_reward = 0

        if self.use_progress_reward: 
            # state[0] is body height above ground, body_rpy[1] is pitch
            self._alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z,
                                                    self.robot.body_rpy[1]))
            done = self._isDone()
            if not np.isfinite(state).all():
                print("~INF~", state)
                done = True

            potential_old = self.potential
            self.potential = self.robot.calc_potential()
            progress = float(self.potential - potential_old)

            total_reward += self._alive
            total_reward += progress

        if self.use_limit_cost: 
            joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
            total_reward += joints_at_limit_cost

        if self.use_strain_cost: 
            sum_strain = 0
            for joint in self.ordered_joints: 
                _, _, forces, _ = joint._p.getJointState(joint.bodies[joint.bodyIndex], joint.jointIndex)
                # sum of moments 
                sum_strain += np.linalg.norm( np.array(forces[3:]))
            if "strain" in self.prev_state_memory: 
                strain_cost =  self.strain_cost * (sum_strain  - self.prev_state_memory["strain"])
                total_reward += strain_cost
            self.prev_state_memory["strain"] = sum_strain

        if self.use_electricity_cost: 
            electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
                ))  # let's assume we have DC motor with controller, and reverse current braking
            electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
            total_reward += electricity_cost


        self.HUD(state, a, done)

        if self.use_electricity_surprise or self.use_strain_surprise:
            current_predict_vals = []

            if self.use_electricity_surprise:
                current_predict_vals.append((self.electricity_surprise_weight, electricity_cost))
            if self.use_strain_surprise:
                current_predict_vals.append((self.strain_surprise_weight, sum_strain))

            for i, current_val_info in enumerate(current_predict_vals):
                surprise_penalty_weight, current_val = current_val_info
                self.ensemble_training_datas[i].append((old_state, a, current_val))

                if len(self.ensemble_training_datas[i]) == 1000:
                    #print('Training deep normal models, 1000 steps passed')
                    for penalty_model in self.penalty_ensembles[i]:
                        penalty_model.train(self.ensemble_training_datas[i])
                    self.ensemble_training_datas[i] = []
                if self.step_counter % 50000 == 0:
                    print(str(self.step_counter) + " steps passed")

                if self.step_counter > 20000 and current_val < 0:
                    #Getting ensemble prediction, lambda_1' and lambda_2' are both set to 2 (a default value the paper gives), but this is adjustable
                    a_lambdaprime = 1/(1 + np.exp(self.lambda1prime*(current_val - self.lambda2prime)))

                    predicted_dists = [model(torch.cat((torch.from_numpy(old_state), torch.from_numpy(a))).float()) for model in self.penalty_ensembles[i]]
                    predicted_dists_mean = torch.mean([dist.mean for dist in predicted_dists], 0)
                    predicted_dists_var = torch.mean([dist.variance + torch.square(dist.mean) for dist in predicted_dists], 0) - torch.square(predicted_dists_mean)
                    neg_log_likelihood = -torch.normal(mean=predicted_dists_mean, std=torch.sqrt(predicted_dists_var)).log_prob(current_val)

                    penalty_based_surprise_reward = a_lambdaprime*(neg_log_likelihood) + (1 - a_lambdaprime)*current_val
                    assert(isinstance(total_reward, float))
                    total_reward += penalty_based_surprise_reward*surprise_penalty_weight
                
        self.reward = total_reward
        return total_reward 

    def step(self, a):
        """Fully overrides `step` in `WalkerBaseBulletEnv` base class."""
        self.step_counter += 1

        old_state = self.robot.calc_state()

        # if multiplayer, action first applied to all robots,
        # then global step() called, then _step() for all robots
        # with the same actions
        if not self.scene.multiplayer:
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        total_reward = self.compute_reward(old_state, a, state)
        return state, total_reward, bool(done), {}
