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

# --- CONSTANTS ---
CTRL_FREQ = 50
SENSOR_FREQ = 100
PLOTTING_FREQ = 30
CTRL_PERIOD = 1/CTRL_FREQ
SENSOR_PERIOD = 1/SENSOR_FREQ
PLOTTING_PERIOD = 1/PLOTTING_FREQ

# --- MODULE VARIABLES ---
last_keystroke = None

start_timestamp  = None
ctrl_timestamp = None
sensor_timestamp = None
plotting_timestamp = None

motor_enc_raw = None
pendulum_enc_raw = None

data_x = [0] * 100
data_y1 = [0] * 100
data_y2 = [0] * 100
counter = 0
curve1 = None
curve2 = None

# --- FUNCTION DEFINITIONS ---
def update_plots():
	global data_x, data_y1, data_y2, counter, curve1, curve2
	y1_val = np.sin(time.time())
	y2_val = np.cos(time.time())

	data_x.append(counter)
	data_x = data_x[1:]

	data_y1.append(y1_val)
	data_y1 = data_y1[1:]

	data_y2.append(y2_val)
	data_y2 = data_y2[1:]

	curve1.setData(data_x, data_y1)
	curve2.setData(data_x, data_y2)

	counter += 1

# --- CLASS DEFINITIONS ---

# --- MAIN FUNCTION ---
def main():
	global last_keystroke, start_timestamp, ctrl_timestamp, sensor_timestamp, plotting_timestep, curve1, curve2

	# Initialize timers
	start_timestamp = time.time()
	ctrl_timestamp = start_timestamp
	sensor_timestamp = start_timestamp
	plotting_timestamp = start_timestamp

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

	print("ODrive configured!")

	# --- LOOP SETUP ---
	# Setup plotting
	app = pg.mkQApp("Real-Time Plotting")
	win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Plotting")
	win.resize(1000, 600)

	plot1 = win.addPlot(title="Real-Time Data")
	curve1 = plot1.plot(pen='r', name="Trace 1")

	win.nextRow()
	plot2 = win.addPlot(title="Real-Time Data")
	curve2 = plot2.plot(pen='r', name="Trace 2")

	plot1.addLegend()
	plot2.addLegend()

	plot_timer = QtCore.QTimer()
	plot_timer.timeout.connect(update_plots)
	plot_timer.start(int(PLOTTING_PERIOD * 1000))
	app.exec_()

	# # Start ODrive control
	# odrv.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL

	# # --- MAIN LOOP ---
	# for i in range(100):
	# 	pos_target = np.sin(time.time())
	# 	odrv.axis0.controller.input_pos = pos_target
	# 	time.sleep(0.03)

	# odrv.axis0.requested_state = AXIS_STATE_IDLE
	# time.sleep(3)
	print("Main function completed...")

# --- TEST CODE ---
if __name__ == "__main__":
    main()

