from pybullet_envs.gym_locomotion_envs import HopperBulletEnv
from model import LearnHopperPenalty
import numpy as np 
import torch

class Hopper(HopperBulletEnv):
  electricity_cost = -0.001  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
  stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
  foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
  foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
  joints_at_limit_cost = -0.1  # discourage stuck joints
  strain_cost = -0.0001
  electricity_surprise_weight = 1
  strain_surprise_weight = 1
  
  def __init__(self, enable_torque, predict_val, add_additional=True, render=False, episode_steps=1000):
    """Modifies `__init__` in `HopperBulletEnv` parent class."""
    self.episode_steps = episode_steps
    self.enable_torque = enable_torque
    self.predict_val = predict_val # either "electricity" or "strain" or "electricity_strain" or ""
    self.add_additional = add_additional
    super().__init__(render=render)

    self.torque_enabled = False

  def reset(self):
    """Modifies `reset` in `WalkerBaseBulletEnv` base class."""
    self.step_counter = 0
    self.ensemble_training_datas = []
    self.penalty_ensembles = []
    if "electricity" in self.predict_val:
      self.ensemble_training_datas.append([])
      self.penalty_ensembles.append([LearnHopperPenalty(seed=idx) for idx in range(5)]) #DeepNormal models go here)
    if "strain" in self.predict_val:
      self.ensemble_training_datas.append([])
      self.penalty_ensembles.append([LearnHopperPenalty(seed=idx) for idx in range(5, 10)]) #DeepNormal models go here)
    ret_val = super().reset()

    if self.enable_torque and not self.torque_enabled: 
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

    feet_collision_cost = 0.0
    for i, f in enumerate(self.robot.feet):
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      if (self.ground_ids & contact_ids):
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0

    joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)

    if self.enable_torque: 
      sum_strain = 0
      for joint in self.ordered_joints: 
          _, _, forces, _ = joint._p.getJointState(joint.bodies[joint.bodyIndex], joint.jointIndex)
          # sum of moments 
          sum_strain += np.linalg.norm( np.array(forces[3:]))
      sum_strain *= self.strain_cost

    electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
    ))  # let's assume we have DC motor with controller, and reverse current braking
    electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

    base_rewards = [
                    self._alive, progress,
                    joints_at_limit_cost, feet_collision_cost, 
                    ]

    if self.enable_torque: 
        additional_reward = [electricity_cost, sum_strain]
    else: 
        additional_reward = [electricity_cost]
    
    if self.add_additional: 
        self.rewards = base_rewards + additional_reward
    else: 
        self.rewards = base_rewards

    self.HUD(state, a, done)
    #print(self.rewards)
    self.reward += sum(base_rewards)
    
    total_reward = sum(self.rewards)
    if self.predict_val:
        current_predict_vals = []
        if "electricity" in self.predict_val:
            current_predict_vals.append((electricity_surprise_weight, electricity_cost))
        if "strain" in self.predict_val:
            current_predict_vals.append((strain_surprise_weight, sum_strain))
        for i, current_val_info in enumerate(current_predict_vals):
            surprise_penalty_weight, current_val = current_val_info
            self.ensemble_training_datas[i].append((old_state, a, predict_val))

            if len(self.ensemble_training_datas[i]) == 1000:
            #print('Training deep normal models, 1000 steps passed')
                for penalty_model in self.penalty_ensembles[i]:
                    penalty_model.train(self.ensemble_training_datas[i])
                self.ensemble_training_datas[i] = []
            if self.step_counter % 50000 == 0:
                print(str(self.step_counter) + " steps passed")

            if self.step_counter > 20000 and current_val < 0:
            #Getting ensemble prediction, lambda_1' and lambda_2' are both set to 2 (a default value the paper gives), but this is adjustable
                lambda1prime = 2
                lambda2prime = 2
                a_lambdaprime = 1/(1 + np.exp(lambda1prime*(current_val - lambda2prime)))

                predicted_dists = [model(torch.cat((torch.from_numpy(old_state), torch.from_numpy(a))).float()) for model in self.penalty_ensembles[i]]
                predicted_dists_mean = torch.mean([dist.mean for dist in predicted_dists], 0)
                predicted_dists_var = torch.mean([dist.variance + torch.square(dist.mean) for dist in predicted_dists], 0) - torch.square(predicted_dists_mean)
                neg_log_likelihood = -torch.normal(mean=predicted_dists_mean, std=torch.sqrt(predicted_dists_var)).log_prob(current_val)

                penalty_based_surprise_reward = a_lambdaprime*(neg_log_likelihood - current_val)
                assert(isinstance(total_reward, float))
                total_reward += penalty_based_surprise_reward*surprise_penalty_weight

    return state, total_reward, bool(done), {}
