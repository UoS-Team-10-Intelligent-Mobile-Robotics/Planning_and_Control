#!/bin/bash

# This script is run by crontab on startup. It is used to start the robot
# software.

# To install it, run the following command:
# sudo crontab -e
# Then add the following line to the end of the file:
# @reboot /home/pi/git/uos_feeg6043_build/crontab_startup.sh

# The command
# $ sudo wifi-ap-sta status
# should return something like this:
# State: active
# Access Point Network SSID:      pi-top-A2ED
# Access Point Wi-Fi Password:    sjp9xyc2
# Access Point IP Address:        192.168.90.1

# Get the wifi-ap-sta status
wifi_status=$(sudo wifi-ap-sta status)
#ssid=$(echo "$wifi_status" | awk -F 'SSID: ' '{print $2}' | awk '{print $1}')
#password=$(echo "$wifi_status" | awk -F 'Password: ' '{print $2}' | awk '{print $1}')
#ip_address=$(echo "$wifi_status" | awk -F 'IP Address: ' '{print $2}' | awk '{print $1}')

# Function to check state
check_state() {
    local state="inactive"
    while [ "$state" != "active" ]; do
        # Your command to check state here, for example:
	state=$(echo $(sudo wifi-ap-sta status) | awk -F 'State: ' '{print $2}' | awk '{print $1}')
        sleep 1
    done
}

adddate() {
    while IFS= read -r line; do
        printf '%s %s\n' "$(date)" "$line";
    done
}

# Wait a bit
sleep 20

# Wait for Wi-Fi AP
check_state
sudo wifi-ap-sta status > /home/pi/.wifi_status

# Pull if connected to internet
cd /home/pi/git/uos_feeg6043_build && git pull origin main | adddate >> /home/pi/logs/pitop_crontab.log
cd /home/pi/git/uos_feeg6043_build && pip install -U . | adddate >> /home/pi/logs/pitop_crontab.log

# Start the robot software
python3 /home/pi/git/uos_feeg6043_build/src/robot.py | adddate >> /home/pi/logs/pitop_crontab.log
