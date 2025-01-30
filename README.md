# ODriveCartPole
ODrive cart pole repo for inverted pendulum lab

## Installation

Nate's installation experience on Ubuntu 22.04:

**Prerequisites:** this tutorial assumes that you have installed the [mamba](https://github.com/mamba-org/mamba) package manager. If you have conda instead, you can replace mamba with conda in the commands below.

```
mamba create -n cartpole
mamba activate cartpole
pip3 install matplotlib
pip3 install --upgrade odrive
mamba install pytorch torchvision torchaudio -c pytorch
pip3 install torchsummary
```

Set up the udev tools: (from [odrive](https://docs.odriverobotics.com/v/latest/interfaces/odrivetool.html))
```
sudo bash -c "curl https://cdn.odriverobotics.com/files/odrive-udev-rules.rules > /etc/udev/rules.d/91-odrive.rules && udevadm control --reload-rules && udevadm trigger"
```

To run the cart-pole:
```
mamba activate cartpole
cd ODriveCartPole
python3 odrive_ctrl_v7.py
```