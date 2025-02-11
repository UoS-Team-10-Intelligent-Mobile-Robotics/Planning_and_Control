---
file_format: mystnb
kernelspec:
  name: python3
---

# PiTop [4] robots

- Username: pi
- Password: pi-top

You must register the mac address on 
https://network.soton.ac.uk/wired/register
to clone the repo onto the robot

## How to work with the pi-top [4]

Turn the robot ON by flicking the power switch on its side for a couple of seconds. The
platform should turn itself on and display similar to below:

Wi-Fi: pi-top-XXXX
Pass: XXXXXXXX
192.168.90.1 v1.05

It automatically runs its local version of robot.py with zero-ROS for communication.

Connect to your robot's Wi-Fi server (you should see pi-top-XXXX on your wireless network). Make sure you do not conne t to a different robot.

Run your laptop.py 

## How to reflash the firmware

1. Go to https://knowledgebase.pi-top.com/knowledge/reflashing-your-sd-card and download the latest pi-topOS image.
2. Burn the image to an SD card using Etcher (https://www.balena.io/etcher/).
3. Insert the SD card into the pi-top [4] and power it on. The pi-top [4] will boot into the pi-topOS setup wizard.
4. Follow the on-screen instructions to complete the setup wizard.
5. Once the setup wizard is complete, you can SSH into the pi-top [4]. The default hostname is `pi-top-XXXXXX` where `XXXXXX` is the last 6 characters of the pi-top [4]'s serial number. The default username and password are `pi` and `pi-top`.

## Install feeg6043 build software
1. Connected to your robot's wifi server 
2. Type http://192.168.90.1/ into a browser to access to its local operating system
3. Open a terminal and go into the directory 'cd ~/git'. If you need to create this first, type 'mkdir ~/git' in your terminal
4. Clone the git repository 'git clone https://oauth2:tbFTQFj2HL8j9ecb1BQe@git.soton.ac.uk/feeg6043/uos_feeg6043_build.git'. If it already exists, pull it with 'git pull https://oauth2:tbFTQFj2HL8j9ecb1BQe@git.soton.ac.uk/feeg6043/uos_feeg6043_build.git'
5. Go into the git repository 'cd uos_feeg6043_build' and install the software with 'pip install -U --user -e .'. If you get a error message that the externally managed environments, run the command with 'pip install -U --user -e . --break-system-packages' flag. 
6. Open crontab 'crontab -e'. Create new crontab directory if needed.
7. Add the following at the bottom of the line '@reboot /home/pi/git/uos_feeg6043_build/crontab_startup.sh'
8. Reboot system and it should open and run 'robot.py' automatically ready for commands from 'laptop.py'

## Update feeg6043 build software
Follow steps 1-5 above.