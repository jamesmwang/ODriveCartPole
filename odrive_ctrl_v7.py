# --- NOTES ---
# In Python, you only need to declare a module variable as global in a function
# if you are modifying its value in the function. If the variable is a class instance
# and you are modifying the instance attributes in the function, you do not need to
# declare it as global

# Logical statements and loops do not introduce a new scope

# Bare bones FPS: main - 1.89M
# ODrive init FPS: main - 1.85M
# Reading raw encoder: main - 1.5M, encoder - 150

# --- IMPORTS ---
import time
import numpy as np
from collections import deque
import odrive
from odrive.enums import *
import socket
import json
import signal

import yaml
import torch
import torch.nn as nn
from torchsummary import summary
from rl_inference import CartPolePolicy

# --- CONSTANTS ---
# ODrive options
MAX_POS_DC_CURRENT = 20
MOTOR_CURRENT_SOFT_MAX = 50
MOTOR_CURRENT_HARD_MAX = 70
AXIS_TORQUE_SOFT_MIN = -2 # Set to low value for testing
AXIS_TORQUE_SOFT_MAX = 2 # Set to low value for testing
AXIS_VEL_LIMIT = 20

# Program options
# CTRL_MODE = "PID"
# CTRL_MODE = "LQR"
CTRL_MODE = "RL"
THETA_LIM = np.radians(20) # Input in degrees
# THETA_LIM = np.inf # For swing up mode
X_LIM = 0.4
FORCE_LIM = 30
BUILT_IN_MOTOR_ENCODER = True
PRINT_FPS = False
FPS_PRINT_FREQ = 0.5
SENSOR_FREQ = 500
CTRL_FREQ = 50
VEL_ALPHA = 0.1

# PID gains (all neg)
# KP = -150
# KI = -1
# KD = -0.4
KP = -100
KI = -0
KD = -10

# LQR gains
# K_X = -22.36067977
# K_X_DOT = -21.52266472
# K_THETA = -83.79167852
# K_THETA_DOT = -15.36174827
# LQR_FACTOR = 1.0 # To increase gains

K_X = -17
K_X_DOT = -20
K_THETA = -97
K_THETA_DOT = -24
LQR_FACTOR = 1.0 # To increase gains

# RL model path
rl_model_path = "rl_model/params_012925_working_checkpoint.yaml"

# UDP transmission/plotting options
UDP_FREQ = 30
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
N_TRACES = 4 # x, theta, x dot, theta dot

# Other
FPS_PRINT_PERIOD = 1/FPS_PRINT_FREQ
SENSOR_PERIOD = 1/SENSOR_FREQ
CTRL_PERIOD = 1/CTRL_FREQ
UDP_PERIOD = 1/UDP_FREQ

# --- MODULE VARIABLES ---
running = True
ctrl_z_flag = False
ctrl_force = 0

# --- FUNCTION DEFINITIONS ---
def sigint_handler(sig, frame):
	global running
	running = False
	print("\nctrl+c detected, exiting main loop...")

def sigstp_handler(signum, frame):
	global ctrl_z_flag
	ctrl_z_flag = True

def torque_to_force(torque, r):
	return torque/r

def force_to_torque(force, r):
	return force*r

# --- CLASS DEFINITIONS ---
class FPSMonitor:
	def __init__(self, N=100):
		self.N = N
		self.timestamps = deque(maxlen=self.N)

	def get_fps(self):
		self.timestamps.append(time.time())
		fps = 0
		if len(self.timestamps) > 1:
			fps = len(self.timestamps)/(self.timestamps[-1] - self.timestamps[0])

		return fps

class AbsoluteEncoder:
	def __init__(self):
		self.last_reading = None
		self.pos = None

	def update_pos(self, reading):
		if self.last_reading is None:
			self.last_reading = reading
			self.pos = reading
			return self.pos

		delta_reading = reading - self.last_reading

		# Handle wrap around condition
		if delta_reading > 0.5: # Wrapped around from ~0 to ~1
			delta_reading -= 1.0
		elif delta_reading < -0.5: # Wrapped around from ~1 to ~0
			delta_reading += 1.0

		self.pos += delta_reading
		self.last_reading = reading
		return self.pos

class CartPole:
	def __init__(self, alpha = 0.25):
		# State vector
		self.last_x = None
		self.x = None
		self.x_dot = None
		self.last_theta = None
		self.theta = None
		self.theta_dot = None
		self.state_timestamp = None

		# Geometry
		self.r_pulley = 0.02865 # m
		self.rail_length = None

		# Filtering
		self.alpha = alpha

		# Zeroing
		self.cart_lim_1 = None
		self.cart_lim_2 = None
		self.cart_zero = None
		self.pendulum_zero = None
		self.zeroed = False
		self.update_last_pos = True

	def convert_x(self, motor_pos, sign=1):
		# Note motor position is in rotations
		return sign*2*np.pi*self.r_pulley*motor_pos

	def convert_theta(self, pendulum_pos, sign=-1):
		# Note pendulum pos is in rotations
		return sign*2*np.pi*pendulum_pos

	def zero_cart(self):
		try:
			if self.cart_lim_1 is not None and self.cart_lim_2 is not None:
				self.cart_zero = (self.cart_lim_1 + self.cart_lim_2)/2
				return self.cart_zero
			else:
				raise ValueError("One or more cart limits is not set...")
		except ValueError as e:
				print(f"ERROR: {e}")
				while(True): time.sleep(1)

	def zero_cart_pole(self):
		try:
			if self.cart_lim_1 is not None and self.cart_lim_2 is not None and self.pendulum_zero is not None:
				self.zeroed = True
				self.rail_length = np.abs(self.cart_lim_1 - self.cart_lim_2)
			else:
				raise ValueError("Cart or pendulum is not zeroed...")
		except ValueError as e:
				print(f"ERROR: {e}")
				while(True): time.sleep(1)

	def update_state(self, motor_pos, pendulum_pos, dt):
		# Convert angular pendulum position to SI displacement with correct sign
		new_theta = self.convert_theta(pendulum_pos, sign=-1)

		# Convert angular motor position to SI displacement with correct sign
		new_x = self.convert_x(motor_pos, sign=1)

		# Update positions
		self.x = new_x
		self.theta = new_theta

		# Update velocities
		if dt > 0 and self.last_x != None and self.last_theta != None:
			# Calculate discrete derivative
			new_x_dot = (new_x - self.last_x)/dt
			new_theta_dot = (new_theta - self.last_theta)/dt

			# Apply exponential filter to x
			if self.x_dot == None:
				self.x_dot = new_x_dot
			else:
				self.x_dot = self.alpha*new_x_dot + (1 - self.alpha)*self.x_dot

			# Apply exponential filter to theta
			if self.theta_dot == None:
				self.theta_dot = new_theta_dot
			else:
				self.theta_dot = self.alpha*new_theta_dot + (1 - self.alpha)*self.theta_dot

		# Update state timestamp
		self.state_timestamp = time.time()

		# Log history for differentiation
		self.last_x = self.x
		self.last_theta = self.theta

	def get_state_vector(self):
		if self.zeroed:
			zeroed_x = self.x - self.cart_zero
			zeroed_theta = self.theta - self.pendulum_zero - np.pi # Subtract pi to account for the fact that pendulum is zeroed while dangling
			return [zeroed_x, zeroed_theta, self.x_dot, self.theta_dot]
		else:
			return [self.x, self.theta, self.x_dot, self.theta_dot]

class PIDController:
	def __init__(self, kp, ki, kd):
		self.kp = kp
		self.ki = ki
		self.kd = kd
		self.integral = 0

	def get_ctrl(self, setpoint, measured_value, derivative, dt):
		error = setpoint - measured_value
		self.integral += error*dt
		# Sign of kd term needs to be negative to account for error definition
		output = self.kp*error + self.ki*self.integral - self.kd*derivative
		return output

class LQRController:
	def __init__(self, k_mat):
		self.k_mat = np.array(k_mat) # [k_x, k_x_dot, k_theta, k_theta_dot]

	def get_ctrl(self, state_vector):
		state_vector = np.array(state_vector)
		# k_mat_mod = np.array([self.k_mat[0], self.k_mat[1], 0, 0])
		# k_mat_mod = np.array([0, 0, self.k_mat[2], self.k_mat[3]])
		k_mat_mod = self.k_mat
		output = -np.dot(state_vector, k_mat_mod)
		return output


class FSM:
	def __init__(self, name, states):
		self.name = name
		self.states = states
		self.state = None

	def switch_state(self, state_request):
		if state_request not in self.states:
			print(f"{self.name} FSM Error: {state_request} state request invalid...")
		else:
			self.state = state_request
			print(f"STATE: {self.state}")

class RuntimeAssurance:
	def __init__(self, force_lim = 10, x_lim=0.4, x_dot_lim=np.inf, theta_lim=0.15, theta_dot_lim=np.inf):
		self.force_lim = force_lim
		self.x_lim = x_lim
		self.x_dot_lim = x_dot_lim
		self.theta_lim = theta_lim
		self.theta_dot_lim = theta_dot_lim

	def check(self, ctrl_force, cart_pole_state):
		x = cart_pole_state[0]
		theta = cart_pole_state[1]
		x_dot = cart_pole_state[2]
		theta_dot = cart_pole_state[3]

		if np.abs(ctrl_force) > self.force_lim:
			print(f"WARNING: {ctrl_force} exceeds control force limit: {self.force_lim}")
			return False
		elif np.abs(x) > self.x_lim:
			print(f"WARNING: {x} exceeds cart position limit: {self.x_lim}")
			return False
		elif np.abs(theta) > self.theta_lim:
			print(f"WARNING: {theta} exceeds pendulum position limit: {self.theta_lim}")
			return False
		elif np.abs(x_dot) > self.x_dot_lim:
			print(f"WARNING: {x_dot} exceeds cart velocity limit: {self.x_dot_lim}")
			return False
		elif np.abs(theta_dot) > self.theta_dot_lim:
			print(f"WARNING: {theta_dot} exceeds pendulum velocity limit: {self.theta_dot_lim}")
			return False
		else:
			return True

# --- MAIN FUNCTION ---
def main():
	global ctrl_z_flag, ctrl_force

	# Register signal handlers
	signal.signal(signal.SIGINT, sigint_handler) # Exit main loop
	signal.signal(signal.SIGTSTP, sigstp_handler) # Interrupt-based user input

	print("Main loop starting...")
	print("Enter 'ctrl+c' to exit program...")

	# --- ODRIVE SETUP ---
	# Connect to ODrive
	print("Connecting to ODrive...")
	try:
		odrv = odrive.find_any(timeout=5)
		print("ODrive connected!")
	except Exception as e:
		print(f"Failed to connect to ODrive: {e}")
		return

	# Configure power source
	# Battery
	if True:
		bat_n_cells = 4
		odrv.config.dc_bus_undervoltage_trip_level = 3.3 * bat_n_cells
		odrv.config.dc_bus_overvoltage_trip_level = 4.25 * bat_n_cells
		odrv.config.dc_max_positive_current = MAX_POS_DC_CURRENT
		odrv.config.dc_max_negative_current = -np.inf
		odrv.config.brake_resistor0.enable = False

	# Configure motor
	odrv.axis0.config.motor.motor_type = MotorType.HIGH_CURRENT
	odrv.axis0.config.motor.pole_pairs = 20
	odrv.axis0.config.motor.torque_constant = 0.0827
	odrv.axis0.config.motor.current_soft_max = MOTOR_CURRENT_SOFT_MAX
	odrv.axis0.config.motor.current_hard_max = MOTOR_CURRENT_HARD_MAX
	odrv.axis0.config.motor.calibration_current = 10
	odrv.axis0.config.motor.resistance_calib_max_voltage = 2
	odrv.axis0.config.calibration_lockin.current = 10
	odrv.axis0.motor.motor_thermistor.config.enabled = False

	# Configure control
	ctrl_select = "TORQUE"

	if ctrl_select == "POSITION":
		odrv.axis0.controller.config.control_mode = ControlMode.POSITION_CONTROL
		odrv.axis0.controller.config.input_filter_bandwidth = CTRL_PERIOD/2
		odrv.axis0.controller.config.input_mode = InputMode.POS_FILTER

	elif ctrl_select == "TORQUE":
		odrv.axis0.controller.config.control_mode = ControlMode.TORQUE_CONTROL

	odrv.axis0.controller.config.input_mode = InputMode.PASSTHROUGH

	# Set limits
	odrv.axis0.controller.config.vel_limit = AXIS_VEL_LIMIT
	odrv.axis0.controller.config.vel_limit_tolerance = 1.2
	odrv.axis0.config.torque_soft_min = AXIS_TORQUE_SOFT_MIN
	odrv.axis0.config.torque_soft_max = AXIS_TORQUE_SOFT_MAX

	# Configure comm
	odrv.can.config.protocol = Protocol.NONE
	odrv.axis0.config.enable_watchdog = False
	odrv.config.enable_uart_a = False

	# Configure encoders
	# Motor feedback
	odrv.axis0.config.load_encoder = EncoderId.ONBOARD_ENCODER0
	odrv.axis0.config.commutation_encoder = EncoderId.ONBOARD_ENCODER0
	# Pendulum
	odrv.rs485_encoder_group1.config.mode = Rs485EncoderMode.ODRIVE_OA1

	print("ODrive configured!")

	# --- PROGRAM SETUP ---
	# Cart pole instance
	cart_pole = CartPole(alpha=VEL_ALPHA)

	# PID controller instance
	pid = PIDController(KP, KI, KD)

	# LQR controller instance
	k_mat = np.array([K_X, K_X_DOT, K_THETA, K_THETA_DOT])*LQR_FACTOR
	lqr = LQRController(k_mat)

	# Initialize RL control model
	if CTRL_MODE == "RL":
		print("Loading RL model...")
		cart_pole_policy = CartPolePolicy(rl_model_path)
		summary(cart_pole_policy.model_pytorch, input_size=(4,))

	# Runtime assurance instance
	if CTRL_MODE == "RL":
		runtime_assurance = RuntimeAssurance(force_lim=FORCE_LIM, x_lim=X_LIM, theta_lim=np.inf)
	else:
		runtime_assurance = RuntimeAssurance(force_lim=FORCE_LIM, x_lim=X_LIM, theta_lim=THETA_LIM)

	# Timing and FPS counting
	main_fps_monitor = FPSMonitor()
	sensor_fps_monitor = FPSMonitor()
	udp_fps_monitor = FPSMonitor()
	ctrl_fps_monitor = FPSMonitor()
	main_fps = 0
	sensor_fps = 0
	udp_fps = 0
	ctrl_fps = 0

	fps_print_timestamp = time.time()
	sensor_timestamp = time.time()
	udp_timestamp = time.time()
	ctrl_timestamp = time.time()

	# Initialize aboslute encoders
	if BUILT_IN_MOTOR_ENCODER == False: motor_abs_encoder = AbsoluteEncoder()
	pendulum_abs_encoder = AbsoluteEncoder()

	# Initialize UDP plotting
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	plot_state_vector = [0]*N_TRACES

	# Initialize control FSM
	ctrl_fsm = FSM("ctrl", ["STANDBY",
							"CART_LIM_1",
							"CART_LIM_2",
							"DETECT_NEUTRAL",
							"ZERO_PENDULUM",
							"SET_VERTICAL",
							"MOTOR_ON",
							"SANDBOX",
							"PID",
							"LQR",
							"RL",
							"RESET"])

	ctrl_fsm.switch_state("STANDBY")
	print("Press ctrl+z to start zeroing process...")

	# --- MAIN LOOP ---
	while(running):
		main_fps = main_fps_monitor.get_fps()

		# Sensor reading timer
		if time.time() - sensor_timestamp > SENSOR_PERIOD:
			sensor_fps = sensor_fps_monitor.get_fps()

			# Read and unwrap encoders
			if BUILT_IN_MOTOR_ENCODER == False:
				motor_enc_raw = odrv.onboard_encoder0.raw
				motor_pos = motor_abs_encoder.update_pos(motor_enc_raw)
			else:
				motor_pos = odrv.axis0.pos_estimate

			pendulum_enc_raw = odrv.rs485_encoder_group1.raw
			pendulum_pos = pendulum_abs_encoder.update_pos(pendulum_enc_raw)
			
			# Update cart-pole state
			dt = time.time() - sensor_timestamp
			cart_pole.update_state(motor_pos, pendulum_pos, dt)

			sensor_timestamp = time.time()

		# Control timer
		if time.time() - ctrl_timestamp > CTRL_PERIOD:
			ctrl_fps = ctrl_fps_monitor.get_fps()

			# --- CONTROL FSM ---
			if ctrl_fsm.state == "STANDBY":
				if ctrl_z_flag:
					ctrl_z_flag = False
					if cart_pole.zeroed == False:
						ctrl_fsm.switch_state("CART_LIM_1")
						print("Move cart to first limit and press ctrl+z...")
					else:
						ctrl_fsm.switch_state("MOTOR_ON")

			elif ctrl_fsm.state == "MOTOR_ON":
				odrv.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
				if CTRL_MODE == "PID":
					ctrl_fsm.switch_state("PID")
				elif CTRL_MODE == "LQR":
					ctrl_fsm.switch_state("LQR")
				elif CTRL_MODE == "RL":
					ctrl_fsm.switch_state("RL")
				print(f"Press ctrl+z to exit {CTRL_MODE}...")

			elif ctrl_fsm.state == "SANDBOX":
				odrv.axis0.controller.input_torque = 0.0
				if ctrl_z_flag:
					ctrl_z_flag = False
					odrv.axis0.requested_state = AXIS_STATE_IDLE
					ctrl_fsm.switch_state("STANDBY")

			elif ctrl_fsm.state == "PID":
				# Get current state
				state_vector = cart_pole.get_state_vector()
				x = state_vector[0]
				theta = state_vector[1]
				x_dot = state_vector[2]
				theta_dot = state_vector[3]

				# Calculate control force
				dt = time.time() - ctrl_timestamp
				ctrl_force = pid.get_ctrl(0, theta, theta_dot, dt)

				# Runtime assurance
				if runtime_assurance.check(ctrl_force, state_vector):
					# Calculate motor torque
					pulley_rad = cart_pole.r_pulley
					ctrl_torque = force_to_torque(ctrl_force, pulley_rad)
					odrv.axis0.controller.input_torque = ctrl_torque
				else:
					odrv.axis0.requested_state = AXIS_STATE_IDLE
					ctrl_fsm.switch_state("RESET")
					print("Press ctrl+z to reset cart-pole...")

				if ctrl_z_flag:
					ctrl_z_flag = False
					odrv.axis0.requested_state = AXIS_STATE_IDLE
					ctrl_fsm.switch_state("RESET")
					print("Press ctrl+z to reset cart-pole...")

			elif ctrl_fsm.state == "LQR":
				# Get current state
				state_vector = cart_pole.get_state_vector()
				x = state_vector[0]
				theta = state_vector[1]
				x_dot = state_vector[2]
				theta_dot = state_vector[3]

				# Reformat state_vector for LQR
				lqr_state_vector = [x, x_dot, theta, theta_dot]

				# Calculate control force
				ctrl_force = lqr.get_ctrl(lqr_state_vector)

				# Runtime assurance
				if runtime_assurance.check(ctrl_force, state_vector):
					# Calculate motor torque
					pulley_rad = cart_pole.r_pulley
					ctrl_torque = force_to_torque(ctrl_force, pulley_rad)
					odrv.axis0.controller.input_torque = ctrl_torque
				else:
					odrv.axis0.requested_state = AXIS_STATE_IDLE
					ctrl_fsm.switch_state("RESET")
					print("Press ctrl+z to reset cart-pole...")

				if ctrl_z_flag:
					ctrl_z_flag = False
					odrv.axis0.requested_state = AXIS_STATE_IDLE
					ctrl_fsm.switch_state("RESET")
					print("Press ctrl+z to reset cart-pole...")

			elif ctrl_fsm.state == "RL":
				# Get current state
				state_vector = cart_pole.get_state_vector()
				x = state_vector[0]
				theta = state_vector[1]
				x_dot = state_vector[2]
				theta_dot = state_vector[3]

				# Reformat state_vector for RL
				theta_adjusted = (theta + np.pi) % (2*np.pi)
				if theta_adjusted > np.pi:
					theta_adjusted -= 2 * np.pi
					
				rl_state_vector = torch.tensor([x, theta_adjusted, x_dot, theta_dot], dtype=torch.float32)

				# Calculate control force
				ctrl_force = cart_pole_policy.inference(rl_state_vector)

				# Runtime assurance
				if runtime_assurance.check(ctrl_force, state_vector):
					# Calculate motor torque
					pulley_rad = cart_pole.r_pulley
					ctrl_torque = force_to_torque(ctrl_force, pulley_rad)
					odrv.axis0.controller.input_torque = ctrl_torque
				else:
					odrv.axis0.requested_state = AXIS_STATE_IDLE
					ctrl_fsm.switch_state("RESET")
					print("Press ctrl+z to reset cart-pole...")

				if ctrl_z_flag:
					ctrl_z_flag = False
					odrv.axis0.requested_state = AXIS_STATE_IDLE
					ctrl_fsm.switch_state("RESET")
					print("Press ctrl+z to reset cart-pole...")

			elif ctrl_fsm.state == "CART_LIM_1":
				if ctrl_z_flag:
					ctrl_z_flag = False
					cart_pole.cart_lim_1 = cart_pole.x
					print(f"Limit 1 set to: {cart_pole.cart_lim_1}...")
					ctrl_fsm.switch_state("CART_LIM_2")
					print("Move cart to second limit and press ctrl+z...")

			elif ctrl_fsm.state == "CART_LIM_2":
				if ctrl_z_flag:
					ctrl_z_flag = False
					cart_pole.cart_lim_2 = cart_pole.x
					print(f"Limit 2 set to: {cart_pole.cart_lim_2}...")
					cart_zero = cart_pole.zero_cart()
					print(f"Cart position zero set to: {cart_zero}...")
					ctrl_fsm.switch_state("DETECT_NEUTRAL")
					print("Move cart to center of track...")

			elif ctrl_fsm.state == "DETECT_NEUTRAL":
				if np.abs(cart_pole.x - cart_pole.cart_zero) < 0.05:
					if np.abs(cart_pole.x_dot) < 0.005 and np.abs(cart_pole.theta_dot) < 0.5:
						print("Neutral position detected...")
						ctrl_fsm.switch_state("ZERO_PENDULUM")
						print("Press ctrl+z to zero pendulum...")

			elif ctrl_fsm.state == "ZERO_PENDULUM":
				if ctrl_z_flag:
					ctrl_z_flag = False
					cart_pole.pendulum_zero = cart_pole.theta
					print(f"Pendulum zero set to: {cart_pole.pendulum_zero}...")
					cart_pole.zero_cart_pole()
					print(f"Rail length is: {cart_pole.rail_length}")
					print("Cart pole zeroed successfully...")
					ctrl_fsm.switch_state("SET_VERTICAL")
					print("Move pendulum into vertical position...")

			elif ctrl_fsm.state == "SET_VERTICAL":
				zeroed_theta = cart_pole.get_state_vector()[1]
				if np.abs(zeroed_theta) < THETA_LIM:
					ctrl_fsm.switch_state("STANDBY")
					print("Press ctrl+z to start control...")

			elif ctrl_fsm.state == "RESET":
				if ctrl_z_flag:
					ctrl_z_flag = False
					ctrl_fsm.switch_state("ZERO_PENDULUM")
					print("Press ctrl+z to zero pendulum...")

			# --- END ---

			ctrl_timestamp = time.time()

		# UDP transmission timer
		if time.time() - udp_timestamp > UDP_PERIOD:
			udp_fps = udp_fps_monitor.get_fps()

			# Get cart pole state vector
			plot_timestamp = cart_pole.state_timestamp
			plot_state_vector = cart_pole.get_state_vector() # Note vector lengths much match

			# Transmit over UDP if state vector is valid
			if None not in plot_state_vector:
				udp_data = {
					"timestamp": plot_timestamp,
					"state_vector": plot_state_vector,
					"ctrl_force": ctrl_force,
				}
				message = json.dumps(udp_data)
				sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

			udp_timestamp = time.time()

		# FPS printing timer
		if time.time() - fps_print_timestamp > FPS_PRINT_PERIOD and PRINT_FPS:
			print(f"Main, sensor, udp, ctrl FPS: {main_fps:.1f}, {sensor_fps:.1f}, {udp_fps:.1f}, {ctrl_fps:.1f}")
			fps_print_timestamp = time.time()

	print("Motor winding down...")
	odrv.axis0.requested_state = AXIS_STATE_IDLE
	time.sleep(1)
	print("Main loop exited...")

# --- TEST CODE ---
if __name__ == "__main__":
	main()