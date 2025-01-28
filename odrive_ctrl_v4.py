# --- NOTES ---
# Launch odrivetool: odrivetool
# Dump errors: dump_errors(odrv0)
# Clear errors: odrv0.clear_errors()
# Exit odrivetool: exit()

# Plotting is in main thread
# Sensing and control are in separate threads

# Functions can access module variables without global keyword but need global keyword to modify
# Loops don't have separate scope like functions do

# --- IMPORTS ---
import odrive
from odrive.enums import *
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import threading
from collections import deque

# --- OPTIONS ---
PLOTTING = False
PRINT_FPS = 10
CTRL_FREQ = 50
SENSOR_FREQ = 100
PLOTTING_FREQ = 25

# --- CONSTANTS ---
CTRL_PERIOD = 1/CTRL_FREQ
SENSOR_PERIOD = 1/SENSOR_FREQ
PLOTTING_PERIOD = 1/PLOTTING_FREQ
PLOT_WIN_SEC = 3
PLOT_WIN_N = int(PLOTTING_FREQ*PLOT_WIN_SEC)
START_TIMESTAMP = time.time()

# --- MODULE VARIABLES ---
input_mode = "run"

# Timestamps for event timing
start_timestamp  = None
ctrl_timestamp = None
sensor_timestamp = None
plotting_timestamp = None

# Timestamps for frequency monitoring
sensor_timestamps = deque(maxlen=100)
ctrl_timestamps = deque(maxlen=100)
plot_timestamps = deque(maxlen=100)

# Motor encoders
motor_enc_raw = 0
motor_pos = 0
pendulum_enc_raw = 0
pendulum_pos = 0

lock = threading.Lock()

# Plotting
time_vect = deque([0]*PLOT_WIN_N, maxlen=PLOT_WIN_N)
theta_vect = deque([0]*PLOT_WIN_N, maxlen=PLOT_WIN_N)
x_vect = deque([0]*PLOT_WIN_N, maxlen=PLOT_WIN_N)
theta_dot_vect = deque([0]*PLOT_WIN_N, maxlen=PLOT_WIN_N)
x_dot_vect = deque([0]*PLOT_WIN_N, maxlen=PLOT_WIN_N)

theta_curve = None
x_curve = None
theta_dot_curve = None
x_dot_curve = None

# --- CLASS DEFINITIONS ---
class AbsoluteEncoder:
	def __init__(self, alpha=1.0):
		self.last_raw_reading = None
		self.last_orientation = 0
		self.orientation = 0
		self.alpha = alpha

	def update_orientation(self, raw_reading):
		if self.last_raw_reading is None:
			self.last_raw_reading = raw_reading
			self.orientation = raw_reading
			self.last_orientation = self.orientation
			return self.orientation

		# Calculate the change in raw readings
		delta_reading = raw_reading - self.last_raw_reading

		# Handle wrap around condition
		if delta_reading > 0.5: # Neg delta, wrapped around from ~0 to ~1
			delta_reading -= 1.0
		elif delta_reading < -0.5: # Pos delta, wrapped around from ~1 to ~0
			delta_reading += 1.0

		# Check for wrap around errors
		if np.abs(delta_reading) >= 0.4:
			print(f"WARN: {delta_reading}")

		# Update absolute orientation
		new_orientation = self.orientation + delta_reading
		self.orientation = self.alpha*new_orientation + (1 - self.alpha)*self.last_orientation

		self.last_orientation = self.orientation
		self.last_raw_reading = raw_reading
		return self.orientation

class CartPole:
	def __init__(self, alpha = 0.25):
		self.alpha = alpha
		self.last_x = None
		self.x = None
		self.x_dot = 0
		self.last_theta = None
		self.theta = None
		self.theta_dot = 0

	def update_state(self, new_x, new_theta, dt):
		self.x = new_x
		self.theta = new_theta

		if dt > 0 and self.last_x != None and self.last_theta != None:
			new_x_dot = (new_x - self.last_x)/dt
			new_theta_dot = (new_theta - self.last_theta)/dt

			if self.x_dot == None:
				self.x_dot = new_x_dot
			else:
				self.x_dot = self.alpha*new_x_dot + (1 - self.alpha)*self.x_dot

			if self.theta_dot == None:
				self.theta_dot = new_theta_dot
			else:
				self.theta_dot = self.alpha*new_theta_dot + (1 - self.alpha)*self.theta_dot

		self.last_x = self.x
		self.last_theta = self.theta

cart_pole = CartPole()

# --- FUNCTION DEFINITIONS ---
def read_input():
	global input_mode

	print("Input thread started...")

	while True:
		user_input = input()
		if user_input == "x":
			input_mode = user_input

def read_encoders(odrv, read_period):
	global motor_enc_raw, pendulum_enc_raw, motor_pos, pendulum_pos, sensor_timestamp, cart_pole

	print("Encoder thread started...")

	# Create absolute encoder instances
	abs_motor_encoder = AbsoluteEncoder()
	abs_pendulum_encoder = AbsoluteEncoder()

	while True:
		if time.time() - sensor_timestamp > read_period:
			with lock:
				# Read raw encoder values
				motor_enc_raw = odrv.onboard_encoder0.raw
				pendulum_enc_raw = odrv.rs485_encoder_group1.raw
				motor_pos = abs_motor_encoder.update_orientation(motor_enc_raw)
				pendulum_pos = abs_pendulum_encoder.update_orientation(pendulum_enc_raw)
				# Update cart pole state
				dt = time.time() - sensor_timestamp
				cart_pole.update_state(motor_pos, pendulum_pos, dt)

			sensor_timestamp = time.time()
			# For tracking execution frequency
			sensor_timestamps.append(sensor_timestamp)

def control_motor(odrv, ctrl_period):
	global ctrl_timestamp

	print("Control thread started...")

	odrv.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
	while True:
		if time.time() - ctrl_timestamp > ctrl_period:
			with lock:
				motor_pos_copy = motor_pos
				pendulum_pos_copy = pendulum_pos

			odrv.axis0.controller.input_pos = pendulum_pos_copy
			# odrv.axis0.controller.input_pos = 0
			ctrl_timestamp = time.time()
			# For tracking execution frequency
			ctrl_timestamps.append(ctrl_timestamp)

def update_plots():
	global time_vect, theta_vect, x_vect, theta_dot_vect, x_dot_vect
	global theta_curve, x_curve, theta_dot_curve, x_dot_curve

	elapsed_time = time.time() - START_TIMESTAMP

	with lock:
		motor_pos_copy = motor_pos
		pendulum_pos_copy = pendulum_pos
		cart_pole_copy = cart_pole

	# Update data deques
	time_vect.append(elapsed_time)
	theta_vect.append(pendulum_pos_copy)
	x_vect.append(motor_pos_copy)
	x_dot_vect.append(cart_pole_copy.x_dot)
	theta_dot_vect.append(cart_pole_copy.theta_dot)

	# Set data for plotting
	theta_curve.setData(time_vect, theta_vect)
	x_curve.setData(time_vect, x_vect)
	theta_dot_curve.setData(time_vect, theta_dot_vect)
	x_dot_curve.setData(time_vect, x_dot_vect)

	# For tracking execution frequency
	plot_timestamps.append(time.time())

def print_frequencies():
	global sensor_timestamps, ctrl_timestamps, plot_timestamps

	print("Framerate thread started...")

	while True:
		print_period = 1/PRINT_FPS
		time.sleep(print_period)
		with lock:
			if len(sensor_timestamps) > 1:
				sensor_freq = len(sensor_timestamps) / (sensor_timestamps[-1] - sensor_timestamps[0])
				print(f"Sensors: {sensor_freq:.2f} Hz")
			if len(ctrl_timestamps) > 1:
				ctrl_freq = len(ctrl_timestamps) / (ctrl_timestamps[-1] - ctrl_timestamps[0])
				print(f"Control: {ctrl_freq:.2f} Hz")
			if len(plot_timestamps) > 1:
				plot_freq = len(plot_timestamps) / (plot_timestamps[-1] - plot_timestamps[0])
				print(f"Plotting: {plot_freq:.2f} Hz")

# --- MAIN FUNCTION ---
def main():
	global last_keystroke, start_timestamp, ctrl_timestamp, sensor_timestamp, plotting_timestep
	global motor_enc_raw, pendulum_enc_raw
	global theta_curve, x_curve, theta_dot_curve, x_dot_curve 

	# Initialize timers
	ctrl_timestamp = START_TIMESTAMP
	sensor_timestamp = START_TIMESTAMP
	plotting_timestamp = START_TIMESTAMP

	print("Main function started...")
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
		odrv.config.dc_max_positive_current = 20
		odrv.config.dc_max_negative_current = -np.inf
		odrv.config.brake_resistor0.enable = False

	# Configure motor
	odrv.axis0.config.motor.motor_type = MotorType.HIGH_CURRENT
	odrv.axis0.config.motor.pole_pairs = 20
	odrv.axis0.config.motor.torque_constant = 0.0827
	odrv.axis0.config.motor.current_soft_max = 50
	odrv.axis0.config.motor.current_hard_max = 70
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
	odrv.axis0.controller.config.vel_limit = 10
	odrv.axis0.controller.config.vel_limit_tolerance = 1.2
	odrv.axis0.config.torque_soft_min = -20
	odrv.axis0.config.torque_soft_max = 20

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
	# Initialize encoder values
	motor_enc_raw = odrv.onboard_encoder0.raw
	pendulum_enc_raw = odrv.rs485_encoder_group1.raw

	print("ODrive configured!")

	# --- PLOTTING ---
	if PLOTTING:
		print("Plotting thread started...")
		# Initialize pyqtgraph object and window
		app = pg.mkQApp("Real-Time Plotting")
		win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Plotting")
		win.resize(1000, 600)

		# Create plots
		plot1 = win.addPlot(title="Position Data")
		plot1.addLegend()
		theta_curve = plot1.plot(pen='r', name="theta")
		x_curve = plot1.plot(pen='b', name="x")

		win.nextRow()
		plot2 = win.addPlot(title="Velocity Data")
		plot2.addLegend()
		theta_dot_curve = plot2.plot(pen='r', name="theta dot")
		x_dot_curve = plot2.plot(pen='b', name="x dot")

		# Update plots
		plot_timer = QtCore.QTimer()
		# Update curve global variables
		plot_timer.timeout.connect(update_plots)
		plot_timer.start(int(PLOTTING_PERIOD * 1000))

	# --- START THREADS ---
	# Start input monitoring thread
	input_thread = threading.Thread(target=read_input)
	input_thread.daemon = True
	input_thread.start()

	# Start encoder reading thread
	encoder_thread = threading.Thread(target=read_encoders, args=(odrv,SENSOR_PERIOD,))
	encoder_thread.daemon = True
	encoder_thread.start()

	# Start motor control thread
	control_thread = threading.Thread(target=control_motor, args=(odrv,CTRL_PERIOD,))
	control_thread.daemon = True
	control_thread.start()

	# Start FPS monitoring thread
	if PRINT_FPS > 0:
		frequency_thread = threading.Thread(target=print_frequencies)
		frequency_thread.daemon = True
		frequency_thread.start()

	print("Exit plot and enter x to exit program...")

	# Start the plotting app
	# Main thread hangs until plotting app exits
	if PLOTTING: app.exec_()

	# while input_mode == "run":
	# 	time.sleep(0.005)

	print("Winding down...")
	odrv.axis0.requested_state = AXIS_STATE_IDLE
	time.sleep(1)
	print("Main function completed...")

# --- TEST CODE ---
if __name__ == "__main__":
	main()

