# -*- coding: utf-8 -*-
"""
Copyright (c) 2023 The uos_sess6072_build Authors.
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import json
import socket
import weakref
from threading import Thread
from typing import List


class ArUcoUDPDriver:
    def __init__(self, params, parent=None, verbose=False):
        if parent is not None:
            self._parent = weakref.ref(parent)
        self.verbose = verbose
        # -- UDP
        self.client = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )

        # -- Enable port reusage
        self.client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # -- Enable broadcasting mode
        self.client.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # self.client.settimeout(params["timeout"])
        self.client.bind(("", params["port"]))
        self.marker_id = params["marker_id"]
        self.th = Thread(target=self.loop, daemon=True)
        self.th.start()

        # -- data recieved
        self.data = None

    def loop(self):
        while True:
            if self.verbose:
                print("waiting for data...")
            try:
                broadcast_data, _ = self.client.recvfrom(4096)
                result = json.loads(broadcast_data)
                self.data = result.get(str(self.marker_id), None)

            except Exception as e:
                print("Got exception trying to recv %s" % e)

    def read(self) -> List[float]:
        if self.data is None:
            return None
        stamp_s = self.data[0]
        x_m = self.data[2]
        y_m = self.data[3]
        z_m = self.data[4]
        roll_rad = self.data[5]
        pitch_rad = self.data[6]
        yaw_rad = self.data[7]

        # reset data
        self.data = None
        return stamp_s, x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad

    def __del__(self):
        self.client.close()
