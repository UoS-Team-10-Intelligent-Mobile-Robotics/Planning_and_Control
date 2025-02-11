import csv
import time
import copy
import math
import sys
from collections import deque
import traceback
import numpy as np
import pygame

from zeroros import Subscriber
from zeroros.messages import LaserScan

import argparse


def draw_scan(scan_msg, screen_radius, surf):
    for range_m, angle_rad in zip(scan_msg.ranges, scan_msg.angles):
        rel_dist = (screen_radius / 2) * (range_m / 3.0)
        if np.isnan(rel_dist): rel_dist = 0.0
        pos = (
            int(screen_radius / 2) + math.trunc(rel_dist * math.cos(angle_rad)),
            int(screen_radius / 2) + math.trunc(rel_dist * math.sin(angle_rad)),
        )
        pygame.draw.circle(surf, [255, 0, 0], pos, 1)


def exit():
    sys.exit(0)


class LidarDraw:
    def __init__(self, simulation):
        self.lidar_msg = None
        self.robot_ip = "192.168.90.1"
        if simulation:
            self.robot_ip = "127.0.0.1"
        self.laserscan_sub = Subscriber(
            "/lidar", LaserScan, self.laserscan_callback, ip=self.robot_ip
        )
        self.main()

    def laserscan_callback(self, msg):
        print("Received lidar message", msg.header.seq)        
        self.lidar_msg = msg

    def main(self):
        pygame.init()
        clock = pygame.time.Clock()

        font = pygame.font.SysFont("Monospace Regular", 15)

        screen_radius = 1000
        screen_size = (screen_radius, screen_radius)
        screen_center = (int(screen_radius / 2), int(screen_radius / 2))
        screen = pygame.display.set_mode(screen_size)

        last_scans = deque([], 1)
        cur_scan = []

        surf = pygame.Surface(screen_size)
        surf = surf.convert()

        try:
            while True:
                dt = clock.tick(60)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Exiting... (QUIT)")
                        exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("Exiting... (ESCAPE)")
                            exit()

                if self.lidar_msg is None:
                    continue
                if len(self.lidar_msg.ranges) > 0:
                    last_scans.append(self.lidar_msg)
                    self.lidar_msg = None

                surf.fill((250,) * 3)

                for angle in range(0, 360, 15):
                    angle_rad = math.radians(angle)
                    endpt = (
                        int(screen_radius / 2)
                        + math.trunc(screen_radius * math.cos(angle_rad)),
                        int(screen_radius / 2)
                        + math.trunc(screen_radius * math.sin(angle_rad)),
                    )

                    pygame.draw.line(surf, [128, 128, 128], screen_center, endpt)

                    text_angle = math.radians(angle + 1)

                    text_pt = (
                        int(screen_radius / 2)
                        + math.trunc(screen_radius * 0.30 * math.cos(text_angle)),
                        int(screen_radius / 2)
                        + math.trunc(screen_radius * 0.30 * math.sin(text_angle)),
                    )

                    text_surf = font.render(str(angle), False, (0, 0, 0))
                    surf.blit(text_surf, text_pt)

                for i in range(1, 4):
                    pygame.draw.circle(
                        surf,
                        [128, 128, 128],
                        screen_center,
                        math.trunc((screen_radius / 2) * (i / 4)),
                        2,
                    )

                    text_pt = (
                        int(screen_radius / 2)
                        + math.trunc((screen_radius / 2) * (i / 4)),
                        int(screen_radius / 2),
                    )

                    text_surf = font.render(str(i) + "m", False, (0, 0, 0))
                    surf.blit(text_surf, text_pt)

                for scan in last_scans:
                    draw_scan(scan, screen_radius, surf)

                screen.blit(surf, (0, 0))
                pygame.display.flip()
        except KeyboardInterrupt:
            print("Exiting... (KeyboardInterrupt)")
        except Exception as e:
            print("Exiting... (Exception: {})".format(e))
            traceback.print_exc()
        finally:
            exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lidar Draw")
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode. Defaults to False",
    )

    LidarDraw(simulation=parser.parse_args().simulation)
