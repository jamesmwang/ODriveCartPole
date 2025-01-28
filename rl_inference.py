# --- IMPORTS ---
import distutils.util
import os
import subprocess
import mujoco
from mujoco import mjx
from datetime import datetime
from etils import epath
import functools
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict
import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from orbax import checkpoint as ocp
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.sac import train as sac
from brax.io import html, mjcf, model
import time

# --- DEFINE SIMULATION ENVIRONMENT AS MUJOCO XML ---
inverted_pendulum_xml = """
<mujoco model="inverted pendulum">
  <compiler inertiafromgeom="true"/>
  <default>
    <joint armature="0" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.4 0.33 0.26 1.0"/>
    <tendon/>
    <motor ctrlrange="-3 3"/>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".6 .8 1" width="512" height="512"/>
  </asset>
  <option gravity="0 0 -9.81" timestep="0.02" />
  <custom>
    <numeric data="10000" name="constraint_stiffness"/>
    <numeric data="10000" name="constraint_limit_stiffness"/>
    <numeric data="0" name="spring_mass_scale"/>
    <numeric data="1" name="spring_inertia_scale"/>
    <numeric data="5" name="solver_maxls"/>
  </custom>
  <size nstack="3000"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" size="0.01 1.0" type="capsule" rgba="0 0 0 1"/>
    <body name="cart" pos="0 0 0">
      <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-0.5 0.5" type="slide" damping="2.0"/>
      <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.05 0.05" type="capsule" rgba="0.25 0.25 0.25 1" density="229.2"/>
      <body name="pole" pos="0 0 0">
        <joint axis="0 1 0" name="hinge" pos="0 0 0" limited="false" type="hinge" damping="0.0015"/>
        <geom fromto="0 0 0 0 0 -0.6" name="cpole" size="0.014 0.014" type="capsule" rgba="0.5 0 0 1" density="596.35"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-10 10" gear="1" joint="slider" name="slide"/>
  </actuator>
</mujoco>
"""

# --- REPORT SYSTEM PROPERTIES ---
# Function to extract mass and moment of inertia for a given body
def get_mass_and_inertia(mj_model, body_name):
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mass = mj_model.body_mass[body_id]
    inertia = mj_model.body_inertia[body_id]  # Inertia tensor components (ixx, iyy, izz)
    return mass, inertia

# --- REWARD FUNCTION DEFINITIONS ---
alpha_f = 0.90
a_max = 10
alpha_g = 0.5
c_g = 0.57
alpha_h = 0.5
alpha_e = 0.75
c_e = 0.05

# Reward function components
# Reward for low control effort
def a_reward(a):
  return jp.exp(-alpha_f * jp.power(a/a_max, 2))

# Reward for being close to center
def x_reward(x):
  return alpha_g + (1 - alpha_g)*jp.exp(-c_g*jp.power(x, 2))

# Reward for vertical position
# Reversed the sign for weird sign convention of xml initialization
def theta_reward(theta):
  return alpha_h + (1 - alpha_h)*(1 - jp.cos(theta))/2

# Reward for low angular velocity
def dtheta_reward(dtheta):
  return alpha_e + (1 - alpha_e)*jp.exp(-c_e*jp.power(dtheta, 2))

# --- CREATE INVERTED PENDULUM ENVIRONMENT ---
class InvertedPendulum(PipelineEnv):
  def __init__(
      self,
      xml_string,
      reset_noise_scale=1e-2,
      **kwargs,
  ):
    mj_model = mujoco.MjModel.from_xml_string(xml_string)
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6

    sys = mjcf.load_model(mj_model)

    physics_steps_per_control_step = 5
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    self._reset_noise_scale = reset_noise_scale

  # Define the reset method to initialize the environment state
  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    # Add random noise to the initial positions and velocities
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )

    # Initialize the pendulum to any angle
    # qpos = qpos.at[1].set(jax.random.uniform(rng1, minval=-0.25, maxval=0.25))

    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )

    # Initialize the pipeline with the positions and velocities
    data = self.pipeline_init(qpos, qvel)

    # Get the initial observations
    obs = self._get_obs(data, jp.zeros(self.sys.nu))
    reward, done, zero = jp.zeros(3)

    # Initialize metrics
    metrics = {
        'reward_action': zero,
        'reward_position': zero,
        'reward_angle': zero,
        'reward_angular_velocity': zero,
        'total_reward': zero
    }

    # Return the initial state
    return State(data, obs, reward, done, metrics)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    # Scale action from [-1,1] to actuator limits
    action_min = self.sys.actuator.ctrl_range[:, 0]
    action_max = self.sys.actuator.ctrl_range[:, 1]
    action = (action + 1) * (action_max - action_min) * 0.5 + action_min

    data0 = state.pipeline_state
    data = self.pipeline_step(data0, action)

    # Get state vector
    obs = self._get_obs(data, action)
    x = obs[0]
    theta = obs[1]
    x_dot = obs[2]
    theta_dot = obs[3]
    this_action = action[0]

    # Define reward function
    r_a = a_reward(this_action)
    r_x = x_reward(x)
    r_theta = theta_reward(theta)
    r_dtheta = dtheta_reward(theta_dot)
    reward = r_a*r_x*r_theta*r_dtheta

    # Define done condition
    done_theta = jp.abs(theta) > 0.2
    done_x = jp.abs(x) >= 0.25
    # done = jp.where(jp.logical_or(done_theta, done_x), 1.0, 0.0)
    done = jp.where(jp.logical_or(0, done_x), 1.0, 0.0)

    # Update metrics
    state.metrics.update(
        reward_action=r_a,
        reward_position=r_x,
        reward_angle=r_theta,
        reward_angular_velocity=r_dtheta,
        total_reward=reward,
    )

    return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

  def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
    """Observe cartpole body position and velocities."""
    return jp.concatenate([data.qpos, data.qvel, data.qfrc_actuator])

# --- MAIN FUNCTION ---
def main():
	# # Make model, data, and renderer
	# mj_model = mujoco.MjModel.from_xml_string(inverted_pendulum_xml)
	# mj_data = mujoco.MjData(mj_model)

	# Instantiate the environment
	env = InvertedPendulum(xml_string=inverted_pendulum_xml)

	# Define the SAC training parameters
	train_fn = functools.partial(
	    sac.train,
	    num_timesteps=0,
	    num_evals=10,
	    reward_scaling = 0.1,
	    episode_length=1000,
	    normalize_observations=True,
	    action_repeat=1,
	    discounting=0.99,
	    learning_rate=0.0003,
	    num_envs=2048,
	    batch_size=256,
	    grad_updates_per_step=32,
	    max_devices_per_host=1,
	    max_replay_size=1_000_000,
	    min_replay_size=-1,
	    seed=1
	    )

	x_data = []
	y_data = []
	ydataerr = []
	times = [datetime.now()]

	def progress(num_steps, metrics):
	  times.append(datetime.now())
	  x_data.append(num_steps)
	  y_data.append(metrics['eval/episode_reward'])
	  ydataerr.append(metrics['eval/episode_reward_std'])

	# Make inference function
	make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

	print(f'time to jit: {times[1] - times[0]}')
	print(f'time to train: {times[-1] - times[1]}')

	# Load model and define inference function
	model_path = "/Users/jameswang/Documents/CartPole/swing_model_8_16.params"
	params = model.load_params(model_path)
	inference_fn = make_inference_fn(params)
	jit_inference_fn = jax.jit(inference_fn)

	# Make single inference and time
	inference_timestamp = time.time()
	rng = jax.random.PRNGKey(0)
	act_rng, rng = jax.random.split(rng)
	# Note state.obs is six elements long:
	# [x, theta, x_dot, theta_dot, _, _]
	state_obs = jp.array([0.008, -0.0, 0.002, -0.004, 0.0, 0.0])
	ctrl, _ = jit_inference_fn(state_obs, act_rng)
	ctrl_dt = time.time() - inference_timestamp
	print(f"Inference complete, (ctrl, dt) = ({ctrl}, {ctrl_dt})")

# --- TEST CODE ---
if __name__ == "__main__":
	main()
	print("rl_inference.py complete...")

