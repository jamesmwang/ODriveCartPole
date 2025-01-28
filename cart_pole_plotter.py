# --- IMPORTS ---
import socket
import json
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
LEGEND = ["x", "theta", "x_dot", "theta_dot"]
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
            elapsed_time = new_timestamp - START_TIME
            time_vect.append(elapsed_time)
            x_vect.append(new_state[0]*100) # Convert to cm
            theta_vect.append(new_state[1])
            x_dot_vect.append(new_state[2]*100) # Convert to cm/s
            theta_dot_vect.append(new_state[3])
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
    win.resize(1000, 600)

    # Create plots
    plot1 = win.addPlot(title="Position Data")
    plot1.addLegend()
    with lock:
        x_curve = plot1.plot(pen='b', name="x (cm)")
        theta_curve = plot1.plot(pen='r', name="theta (rad)")

    win.nextRow()
    plot2 = win.addPlot(title="Velocity Data")
    plot2.addLegend()
    with lock:
        x_dot_curve = plot2.plot(pen='b', name="x dot (cm/s)")
        theta_dot_curve = plot2.plot(pen='r', name="theta dot (rad/s)")

    win.nextRow()
    plot3 = win.addPlot(title="Control Force")
    plot3.addLegend()
    with lock:
        ctrl_force_curve = plot3.plot(pen='r', name="ctrl force (N)")
        
    # Update plots
    plot_timer = QtCore.QTimer()

    # Update curve global variables
    plot_timer.timeout.connect(update_plots)
    plot_timer.start(int(PLOT_PERIOD * 1000))

    # Executing plotting app hangs main thread
    app.exec_()

if __name__ == "__main__":
    main()
