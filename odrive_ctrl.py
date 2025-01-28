# --- NOTES ---
# Launch odrivetool: odrivetool
# Dump errors: dump_errors(odrv0)
# Clear errors: odrv0.clear_errors()
# Exit odrivetool: exit()

# --- IMPORTS ---
import odrive
from odrive.enums import *
import math
import time
import threading
import numpy as np

# --- CONSTANTS ---
CTRL_FREQ = 10 # Hz
CTRL_PERIOD = 1/CTRL_FREQ # sec
SENSOR_PERIOD = CTRL_PERIOD/2 # sec
START_TIME = time.time() # sec

# --------- ODRIVE SETUP ---------
# --- CONNECT TO ODRIVE ---
print("Connecting to ODrive...")
try:
	odrv = odrive.find_any(timeout=5)
	print("ODrive connected!")
except Exception as e:
	print(f"Failed to connect to ODrive: {e}")

# --- CONFIGURE POWER SOURCE ---
# Battery power source
bat_n_cells = 3
odrv.config.dc_bus_undervoltage_trip_level = 3.3 * bat_n_cells
odrv.config.dc_bus_overvoltage_trip_level = 4.25 * bat_n_cells
odrv.config.dc_max_positive_current = 20
odrv.config.dc_max_negative_current = -math.inf
odrv.config.brake_resistor0.enable = False

# DC power supply source
if False:
	odrv.config.dc_bus_overvoltage_trip_level = 0
	odrv.config.dc_max_positive_current = 0
	odrv.config.dc_max_negative_current = -0.01
	odrv.config.brake_resistor0.resistance = 0
	odrv.config.brake_resistor0.enable = True
	odrv.clear_errors()

# --- CONFIGURE MOTOR ---
odrv.axis0.config.motor.motor_type = MotorType.HIGH_CURRENT
odrv.axis0.config.motor.pole_pairs = 20
odrv.axis0.config.motor.torque_constant = 0.0827
odrv.axis0.config.motor.current_soft_max = 50
odrv.axis0.config.motor.current_hard_max = 70
odrv.axis0.config.motor.calibration_current = 10
odrv.axis0.config.motor.resistance_calib_max_voltage = 2
odrv.axis0.config.calibration_lockin.current = 10
odrv.axis0.motor.motor_thermistor.config.enabled = False

# --- CONFIGURE CONTROL ---
ctrl_select = "POSITION"

if ctrl_select == "POSITION":
	odrv.axis0.controller.config.control_mode = ControlMode.POSITION_CONTROL
	odrv.axis0.controller.config.input_filter_bandwidth = CTRL_PERIOD/2
	odrv.axis0.controller.config.input_mode = InputMode.POS_FILTER

elif ctrl_select == "TORQUE":
	odrv.axis0.controller.config.control_mode = ControlMode.TORQUE_CONTROL

odrv.axis0.controller.config.input_mode = InputMode.PASSTHROUGH

# --- LIMITS ---
odrv.axis0.controller.config.vel_limit = 10
odrv.axis0.controller.config.vel_limit_tolerance = 1.2
odrv.axis0.config.torque_soft_min = -20
odrv.axis0.config.torque_soft_max = 20

# --- CONFIGURE COMM ---
odrv.can.config.protocol = Protocol.NONE
odrv.axis0.config.enable_watchdog = False
odrv.config.enable_uart_a = False

# --- CONFIGURE ENCODERS ---
# Feedback encoder
odrv.axis0.config.load_encoder = EncoderId.ONBOARD_ENCODER0
odrv.axis0.config.commutation_encoder = EncoderId.ONBOARD_ENCODER0
# Secondary encoder
odrv.rs485_encoder_group1.config.mode = Rs485EncoderMode.ODRIVE_OA1

print("ODrive configured!")

# --------- PROGRAM SETUP ---------
# --- MODULE VARIABLES ---
last_keystroke = ""
ctrl_timestamp = START_TIME
sensor_timestamp = START_TIME

# --- FUNCTION DEFINITIONS ---
# Check for user input in separate thread to avoid blocking
def watch_input():
	global last_keystroke
	while last_keystroke != "s":
		user_input = input()
		if user_input == "s":
			last_keystroke = "s"

# --- PENDULUM FSM ---
class PendulumFSM:
	def __init__(self, odrive, active_mode="SANDBOX", ctrl_mode=None):
		self.odrive = odrive
		self.active_mode = active_mode
		self.ctrl_mode = ctrl_mode
		self.states = ["STANDBY", "ACTIVE"]
		self.state = "STANDBY"
		self.user_input = None
		self.motor_enc_reading = None
		self.pendulum_enc_reading = None

	def print_state(self):
		print(f"State: {self.state}")

	# Reads sensors
	def read_sensors(self):
		self.motor_enc_reading = self.odrive.onboard_encoder0.raw
		self.pendulum_enc_reading = self.odrive.rs485_encoder_group1.raw

	# Callback function for plotting
	def get_var_callback(self):
		return [self.motor_enc_reading, self.pendulum_enc_reading]

	# Runs the FSM every control loop
	def run_fsm(self):
		# STANDBY: Motor disabled
		if self.state == "STANDBY":
			if self.user_input == "g":
				self.odrive.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
				self.state = "ACTIVE"
				self.print_state()

		# ACTIVE: Closed loop control for SANDBOX, PID, LQR, or RL active_mode
		elif self.state == "ACTIVE":
			# For experimentation
			if self.active_mode == "SANDBOX":
				if self.ctrl_mode == "POSITION":
					self.odrive.axis0.controller.input_pos = 0

			if self.user_input == "s":
				self.odrive.axis0.requested_state = AXIS_STATE_IDLE
				self.state = "STANDBY"
				self.print_state()

# --- SETUP ---
# Initialize finite state machine
fsm = PendulumFSM(odrive=odrv, active_mode="SANDBOX", ctrl_mode=ctrl_select)
print("Enter g to go...")

# Start program with keystroke
while(last_keystroke != "g"):
	user_input = input()
	if user_input == "g":
		last_keystroke = "g"

# Start thread to monitor for new keystroke
input_thread = threading.Thread(target=watch_input)
input_thread.start()

print("Program started!")
print("Enter s to stop...")

# Enter standby mode
fsm.state = "STANDBY"
print(f"State: {fsm.state}")

# --- MAIN FUNCTION ---
def main():
	global last_keystroke, ctrl_timestamp, sensor_timestamp

	# Plot data
	print("Staring plotter...")
	odrive.utils.start_liveplotter(fsm.get_var_callback, legend=["Motor Encoder", "Pendulum Encoder"])

	while(last_keystroke == "g"):
		elapsed_time = time.time() - START_TIME

		# Check for sensor timer experation
		if time.time() - sensor_timestamp >= SENSOR_PERIOD:
			fsm.read_sensors()
			sensor_timestamp = time.time()

		# Check for control timer experation
		if time.time() - ctrl_timestamp >= CTRL_PERIOD:
			# Run finite state machine
			fsm.user_input = last_keystroke
			fsm.run_fsm()

			ctrl_timestamp = time.time()

	# Finish FSM tasks
	fsm.user_input = last_keystroke
	fsm.run_fsm()

	# Let motor wind down
	print("Winding down...")
	time.sleep(3)

	print("Program ended!")
	input_thread.join()
	print("Input thread closed")

if __name__ == "__main__":
	main()