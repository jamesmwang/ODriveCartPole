import socket
import time
import json
import random

# Settings
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
FREQUENCY = 25  # Hz
N_TRACES = 4  # Number of data traces

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Transmitting data...")

while True:
    data = {
        'timestamp': time.time(),
        'traces': [random.random() for _ in range(N_TRACES)]
    }
    message = json.dumps(data)
    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
    time.sleep(1 / FREQUENCY)
