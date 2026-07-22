# --- IMPORTS ---
import socket
import json
import math
import threading
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from collections import deque
import copy
import time

# --- CONSTANTS ---
# Options
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
PLOT_FREQ = 30
PLOT_WIN_SEC = 3

# Other
PLOT_WIN_N = int(PLOT_FREQ*PLOT_WIN_SEC)
PLOT_PERIOD = 1/PLOT_FREQ
START_TIME = time.time()

# --- MODULE VARIABLES ---
new_data_flag = False

# Incoming data
new_timestamp = None
new_state = None
new_ctrl_force = None

# Plotting data
time_vect = deque(maxlen=PLOT_WIN_N)
theta_vect = deque(maxlen=PLOT_WIN_N)
x_vect = deque(maxlen=PLOT_WIN_N)
theta_dot_vect = deque(maxlen=PLOT_WIN_N)
x_dot_vect = deque(maxlen=PLOT_WIN_N)
ctrl_force_vect = deque(maxlen=PLOT_WIN_N)

# Plotting curves
theta_curve = None
x_curve = None
theta_dot_curve = None
x_dot_curve = None
ctrl_force_curve = None

lock = threading.Lock()

# --- FUNCTION DEFINITIONS ---
def receive_data():
    global new_timestamp, new_state, new_ctrl_force, new_data, new_data_flag

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"Listening on {UDP_IP}:{UDP_PORT}...")

    while True:
        data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
        message = data.decode()
        udp_data = json.loads(message)
        with lock:
            new_timestamp = udp_data["timestamp"]
            new_state = udp_data["state_vector"]
            new_ctrl_force = udp_data["ctrl_force"]
            new_data_flag = True

        # print(f"Message from {addr}:")
        # timestamp = udp_data["timestamp"]
        # print(f"timestamp: {timestamp}, type: {type(timestamp)}")
        # data = udp_data["state_vector"]
        # print(f"data: {data}, type: {type(data)}")

def update_plots():
    global new_data, new_timestamp, new_state, new_ctrl_force, new_data_flag
    global time_vect, theta_vect, x_vect, theta_dot_vect, x_dot_vect, ctrl_force_vect
    global theta_curve, x_curve, theta_dot_curve, x_dot_curve, ctrl_force_curve

    with lock:
        if new_data_flag:
            new_data_flag = False
            # Update data deques
            # Each signal has its own plot, so display units are chosen for
            # readability rather than to make them share an axis
            elapsed_time = new_timestamp - START_TIME
            time_vect.append(elapsed_time)
            x_vect.append(new_state[0]*100) # m -> cm
            theta_vect.append(math.degrees(new_state[1])) # rad -> deg
            x_dot_vect.append(new_state[2]*100) # m/s -> cm/s
            theta_dot_vect.append(math.degrees(new_state[3])) # rad/s -> deg/s
            ctrl_force_vect.append(new_ctrl_force)

            # Set data for plotting
            theta_curve.setData(time_vect, theta_vect)
            x_curve.setData(time_vect, x_vect)
            theta_dot_curve.setData(time_vect, theta_dot_vect)
            x_dot_curve.setData(time_vect, x_dot_vect)
            ctrl_force_curve.setData(time_vect, ctrl_force_vect)

# --- MAIN FUNCTION ---
def main():
    global theta_curve, x_curve, theta_dot_curve, x_dot_curve, ctrl_force_curve

    # Start the receiving thread
    recv_thread = threading.Thread(target=receive_data)
    recv_thread.daemon = True
    recv_thread.start()

    print("Receiver is running...")

    print("Plotting started...")
    # Initialize pyqtgraph object and window
    app = pg.mkQApp("Real-Time Plotting")
    win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Plotting")
    win.resize(1100, 760)

    # One signal per plot. Sharing an axis between a cart position in the tens
    # and a pole angle in the tenths flattened the angle into a straight line,
    # so each signal now gets its own y scale
    plot_defs = [
        ("theta",     0, 0, 1, "Pole angle",     "θ [deg]",       'r'),
        ("x",         0, 1, 1, "Cart position",  "x [cm]",             'b'),
        ("theta_dot", 1, 0, 1, "Pole rate",      "θ̇ [deg/s]", 'r'),
        ("x_dot",     1, 1, 1, "Cart velocity",  "ẋ [cm/s]",      'b'),
        ("force",     2, 0, 2, "Control force",  "F [N]",              'y'),
    ]

    # Note the time axes are left to auto-range rather than being linked with
    # setXLink. Every plot draws the same time vector so they line up anyway,
    # and linking views of unequal width matches pixels per unit instead of
    # range, which stretched the double-width force plot to twice the span
    curves = {}
    with lock:
        for key, row, col, colspan, title, ylabel, pen in plot_defs:
            plot = win.addPlot(row=row, col=col, colspan=colspan, title=title)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setLabel("left", ylabel)
            plot.setLabel("bottom", "time [s]")
            curves[key] = plot.plot(pen=pg.mkPen(pen, width=2))

        theta_curve = curves["theta"]
        x_curve = curves["x"]
        theta_dot_curve = curves["theta_dot"]
        x_dot_curve = curves["x_dot"]
        ctrl_force_curve = curves["force"]


    # Update plots
    plot_timer = QtCore.QTimer()

    # Update curve global variables
    plot_timer.timeout.connect(update_plots)
    plot_timer.start(int(PLOT_PERIOD * 1000))

    # Executing plotting app hangs main thread
    # Note pg.exec() works across Qt bindings, PyQt6 dropped app.exec_()
    pg.exec()

if __name__ == "__main__":
    main()
