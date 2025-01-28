# plotter.py
import socket
import json
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from collections import deque
import threading
import time

# Settings
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
PLOTTING_FREQ = 25  # Hz
PLOT_WIN_SEC = 3
PLOT_WIN_N = int(PLOTTING_FREQ * PLOT_WIN_SEC)
N_TRACES = 4  # Number of data traces

# Data storage
time_vect = deque([0] * PLOT_WIN_N, maxlen=PLOT_WIN_N)
data_vects = [deque([0] * PLOT_WIN_N, maxlen=PLOT_WIN_N) for _ in range(N_TRACES)]

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# PyQtGraph setup
app = pg.mkQApp("Real-Time Plotting")
win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Plotting")
win.resize(1000, 600)

plots = []
curves = []
for i in range(N_TRACES):
    plot = win.addPlot(title=f"Data Trace {i+1}")
    plot.addLegend()
    curve = plot.plot(pen=pg.intColor(i), name=f"Trace {i+1}")
    plots.append(plot)
    curves.append(curve)
    win.nextRow()

# Thread lock
lock = threading.Lock()

def update_plots():
    global time_vect, data_vects, curves

    with lock:
        for i in range(N_TRACES):
            curves[i].setData(time_vect, data_vects[i])

def receive_data():
    global time_vect, data_vects

    while True:
        data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        data = json.loads(data.decode())
        timestamp = data['timestamp']
        traces = data['traces']

        with lock:
            time_vect.append(time.time() - timestamp)  # Adjusted for delay
            for i in range(N_TRACES):
                data_vects[i].append(traces[i])

# Start data receiving thread
recv_thread = threading.Thread(target=receive_data)
recv_thread.daemon = True
recv_thread.start()

# Update plots at the specified frequency
plot_timer = QtCore.QTimer()
plot_timer.timeout.connect(update_plots)
plot_timer.start(int(1000 / PLOTTING_FREQ))

if __name__ == "__main__":
    print("Plotting program started...")
    QtWidgets.QApplication.instance().exec_()
