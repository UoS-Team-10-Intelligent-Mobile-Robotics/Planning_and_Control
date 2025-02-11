import time
import re
import datetime
from pbr.version import VersionInfo

import zeroros
from zeroros.messages import Vector3Stamped, LaserScan
from zeroros.message_broker import MessageBroker

from drivers.pitop_controller import PitopController
from drivers.rplidar import RPLidar

import argparse
import numpy as np


def parse_wifi_status(file_path):
    ssid_pattern = re.compile(r"Access Point Network SSID:\s*(.*)")
    password_pattern = re.compile(r"Access Point Wi-Fi Password:\s*(.*)")
    ip_pattern = re.compile(r"Access Point IP Address:\s*(.*)")

    ssid = None
    password = None
    ip_address = None

    with open(file_path, "r") as file:
        for line in file:
            ssid_match = ssid_pattern.match(line)
            if ssid_match:
                ssid = ssid_match.group(1)

            password_match = password_pattern.match(line)
            if password_match:
                password = password_match.group(1)

            ip_match = ip_pattern.match(line)
            if ip_match:
                ip_address = ip_match.group(1)

    return ssid, password, ip_address


class PitopZeroROS:
    def __init__(self, ip="*"):
        # chassis setup
        self.wheel_separation = 0.163
        self.wheel_diameter = 0.065
        self.true_wheel_update_freq = 5  # Hz
        self.lidar_update_period = 0.01  # seconds
        print("Using IP address: " + ip + " for ZeroROS broker")

        # Create ZeroROS broker
        self.broker = MessageBroker(ip=ip)
        # Instance the PiTop driver
        self.controller = PitopController(
            wheel_diameter=self.wheel_diameter, wheel_separation=self.wheel_separation
        )
        self.miniscreen = self.controller.miniscreen

        # Create twist subscriber
        self.wheel_speeds_cmd_sub = zeroros.Subscriber(
            "/wheel_speeds_cmd", Vector3Stamped, self.wheel_speed_callback
        )
        self.true_wheel_speeds_pub = zeroros.Publisher(
            "/true_wheel_speeds", Vector3Stamped
        )
        self.lidar_pub = zeroros.Publisher("/lidar", LaserScan)
        self.last_time_msg_received = None

        self.lidar = RPLidar(
            self.lidar_pub,
            {
                "port": "/dev/ttyUSB0",
                "range_min_m": 0.05,
                "range_max_m": 1.0,
                "angle_min_rad": -np.pi / 3.0,
                "angle_max_rad": np.pi / 3.0,
            },
        )

    def wheel_speed_callback(self, wheel_speed_message):
        wheel_speed_right = wheel_speed_message.vector.x
        wheel_speed_left = wheel_speed_message.vector.y
        print("Received wheel speeds: ", wheel_speed_left, wheel_speed_right, "rad/s")

        self.last_time_msg_received = datetime.datetime.utcnow().timestamp()

        if wheel_speed_left > self.controller.max_motor_speed:
            print("Received speed above max, capping it to max speed")
            wheel_speed_left = self.controller.max_motor_speed
        if wheel_speed_right > self.controller.max_motor_speed:
            print("Received speed above max, capping it to max speed")
            wheel_speed_right = self.controller.max_motor_speed
        self.controller.robot_move(wheel_speed_left, wheel_speed_right)

    def run(self):
        wifi_status_file = "/home/pi/.wifi_status"
        ssid, password, ip_address = parse_wifi_status(wifi_status_file)
        try:
            r = zeroros.Rate(self.true_wheel_update_freq)
            info = VersionInfo("uos_feeg6043_build")
            info.version_string()
            version = info.version_string()
            self.miniscreen.display_multiline_text(
                "Wi-Fi: "
                + ssid
                + " Pass: "
                + password
                + " "
                + ip_address
                + " v"
                + version,
                font_size=12,
            )
            while not self.miniscreen.cancel_button.is_pressed:
                self.true_wheel_speeds_timer_callback()
                r.sleep()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")
        except Exception as e:
            print("Exception: ", e)
        finally:
            self.miniscreen.display_multiline_text("Bye!")
            self.stop()

    def true_wheel_speeds_timer_callback(self):
        # Publish true wheel speeds
        wheel_speed_left, wheel_speed_right = self.controller.current_speed()

        current_time = datetime.datetime.utcnow().timestamp()
        dt = 0.0
        if self.last_time_msg_received is not None:
            dt = current_time - self.last_time_msg_received
        if (wheel_speed_left != 0 or wheel_speed_right != 0) and dt > 3.0:
            print("Wheel speed timeout:", dt, "seconds. Stopping the wheels.")
            self.controller.robot_move(0, 0)

        wheel_speeds_msg = Vector3Stamped()
        wheel_speeds_msg.vector.x = wheel_speed_right
        wheel_speeds_msg.vector.y = wheel_speed_left
        self.true_wheel_speeds_pub.publish(wheel_speeds_msg)

    def stop(self):
        self.wheel_speeds_cmd_sub.stop()
        self.lidar.stop()
        self.controller.robot_move(0.0, 0.0)
        self.broker.stop()


def main():
    print("Starting PitopZeroROS")

    parser = argparse.ArgumentParser(
        description="PitopZeroROS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ip",
        type=str,
        default="*",
        help="IP address of the broker. Defaults to * (all interfaces)",
    )
    args = parser.parse_args()

    ptzr = PitopZeroROS(args.ip)
    ptzr.run()


if __name__ == "__main__":
    main()
