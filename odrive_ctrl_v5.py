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

# --- CONSTANTS ---
# ODrive options
MAX_POS_DC_CURRENT = 20
MOTOR_CURRENT_SOFT_MAX = 50
MOTOR_CURRENT_HARD_MAX = 70
AXIS_TORQUE_SOFT_MIN = -4 # Set to low value for testing
AXIS_TORQUE_SOFT_MAX = 4 # Set to low value for testing
AXIS_VEL_LIMIT = 10

# Program options
BUILT_IN_MOTOR_ENCODER = True
FPS_PRINT_FREQ = 5
SENSOR_FREQ = 500
CTRL_FREQ = 50
VEL_ALPHA = 0.25

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

# --- FUNCTION DEFINITIONS ---
def signal_handler(sig, frame):
	global running
	print("\nInterrupt detected, exiting main loop...")
	running = False

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
		self.alpha = alpha
		self.last_x = None
		self.x = None
		self.x_dot = None
		self.last_theta = None
		self.theta = None
		self.theta_dot = None
		self.state_timestamp = None

	def update_state(self, new_x, new_theta, dt):
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

# --- MAIN FUNCTION ---
def main():
	# Register signal handler to exit main loop
	signal.signal(signal.SIGINT, signal_handler)

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
		bat_n_cells = 3
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
	ctrl_select = "POSITION"

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

	# Start closed loop control
	odrv.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL

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

			odrv.axis0.controller.input_pos = cart_pole.theta

			ctrl_timestamp = time.time()

		# UDP transmission timer
		if time.time() - udp_timestamp > UDP_PERIOD:
			udp_fps = udp_fps_monitor.get_fps()

			# Get cart pole state vector
			plot_timestamp = cart_pole.state_timestamp
			plot_state_vector[0] = cart_pole.x
			plot_state_vector[1] = cart_pole.theta
			plot_state_vector[2] = cart_pole.x_dot
			plot_state_vector[3] = cart_pole.theta_dot

			# Transmit over UDP if state vector is valid
			if None not in plot_state_vector:
				udp_data = {
					"timestamp": plot_timestamp,
					"state_vector": plot_state_vector,
				}
				message = json.dumps(udp_data)
				sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

			udp_timestamp = time.time()

		# FPS printing timer
		if time.time() - fps_print_timestamp > FPS_PRINT_PERIOD:
			print(f"Main, sensor, udp, ctrl FPS: {main_fps:.1f}, {sensor_fps:.1f}, {udp_fps:.1f}, {ctrl_fps:.1f}")
			fps_print_timestamp = time.time()

	print("Motor winding down...")
	odrv.axis0.requested_state = AXIS_STATE_IDLE
	time.sleep(1)
	print("Main loop exited...")

# --- TEST CODE ---
if __name__ == "__main__":
	main()