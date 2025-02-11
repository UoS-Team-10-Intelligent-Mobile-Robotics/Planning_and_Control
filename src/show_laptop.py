"""
Copyright (c) 2023 The uos_sess6072_build Authors.
Authors: Sam Fenton, Blair Thornton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import time
import sys
from threading import Thread
from time import sleep
import argparse
import numpy as np
import copy


from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout
from pglive.sources.data_connector import DataConnector
from pglive.sources.live_plot import LiveLinePlot
from pglive.sources.live_plot import LiveScatterPlot
from pglive.sources.live_plot_widget import LivePlotWidget
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from laptop import LaptopPilot #imports LaptopPilot from `./laptop.py`. If you want to run another version, change laptop on this line to whatever you have called the file you want to run. You don't need to include the py

class Window(QWidget):
    running = False

    def __init__(self, simulation=False, parent=None):
        super().__init__(parent)
        self.wheelrate_plot = LivePlotWidget()
        self.heading_plot = LivePlotWidget()
        self.position_plot = LivePlotWidget()
        # self.timeplot = LivePlotWidget()
        layout = QGridLayout(self)
        layout.addWidget(self.wheelrate_plot, 0, 3, 1, 2)
        layout.addWidget(self.heading_plot, 1, 3, 1, 2)
        layout.addWidget(self.position_plot, 0, 0, 2, 2)
        
        self.loopcounter = 0
        self.Laptop = LaptopPilot(simulation)
        
        # Create one curve pre dataset
        cmd_wheelrate_right = LiveLinePlot(pen="red", name = 'Right Wheel Cmd')
        cmd_wheelrate_left = LiveLinePlot(pen="blue", name = 'Left Wheel Cmd')
        measured_wheelrate_right = LiveLinePlot(pen="magenta", name='Right Wheel Measured')
        measured_wheelrate_left = LiveLinePlot(pen="cyan", name='Left Wheel Measured')
        
        est_heading = LiveLinePlot(pen = 'blue', name = 'Estimated Heading')
        measured_heading = LiveLinePlot(symbol = 'x', pen = 'green', name = 'Measured Heading')       
        
        est_position = LiveScatterPlot(symbol = 'o', size = 4, pen = 'blue', name = 'Estimated Position')
        measured_position = LiveScatterPlot(symbol = 'x', pen = 'green', name = 'Measured Position')
        waypoints = LiveScatterPlot(symbol = 'o', pen = 'red', name = 'Waypoints')
        lidar = LiveScatterPlot(symbol = 'o', size = 1, pen = 'w', name = 'Lidar')
        
        # Data connectors for each plot with dequeue of 600 points
        self.cmd_wheelrate_right = DataConnector(cmd_wheelrate_right, max_points=1500)
        self.cmd_wheelrate_left = DataConnector(cmd_wheelrate_left, max_points=1500)
        self.measured_wheelrate_right = DataConnector(measured_wheelrate_right, max_points=1500)
        self.measured_wheelrate_left = DataConnector(measured_wheelrate_left, max_points=1500)
        
        self.est_heading = DataConnector(est_heading, max_points=1000)
        self.measured_heading = DataConnector(measured_heading, max_points=100)
        
        self.est_position = DataConnector(est_position, max_points=1000)
        self.measured_position = DataConnector(measured_position, max_points=100)
        self.waypoints = DataConnector(waypoints, max_points=50)
        self.lidar = DataConnector(lidar, max_points=3000)
        
        # Show grid
        self.wheelrate_plot.showGrid(x=True, y=True, alpha=0.3)
        self.heading_plot.showGrid(x=True, y=True, alpha=0.3)
        self.position_plot.setAspectLocked()
        self.position_plot.showGrid(x = True, y = True, alpha = 0.3)        

        # Set labels
        self.heading_plot.setLabel('bottom', 'Time', units="s")
        self.heading_plot.setLabel('left', 'Heading', units="degrees")
        self.heading_plot.addLegend()

        self.wheelrate_plot.setLabel('bottom', 'Time', units="s")
        self.wheelrate_plot.setLabel('left', 'Wheel rate', units="rad/s")
        self.wheelrate_plot.addLegend()        
        
        self.position_plot.setLabel('bottom', 'Eastings', units="m")
        self.position_plot.setLabel('left', 'Northings', units="m")
        self.position_plot.addLegend()        

        # Add all three curves
        self.wheelrate_plot.addItem(cmd_wheelrate_right)
        self.wheelrate_plot.addItem(cmd_wheelrate_left)
        self.wheelrate_plot.addItem(measured_wheelrate_right)
        self.wheelrate_plot.addItem(measured_wheelrate_left)
        self.heading_plot.addItem(measured_heading)
        self.heading_plot.addItem(est_heading)
        self.position_plot.addItem(est_position)
        self.position_plot.addItem(measured_position)
        self.position_plot.addItem(waypoints)
        self.position_plot.addItem(lidar)       



    def update(self):
        northings_path = None
        eastings_path = None  
        path_first_counter = None   
        map_x = []
        map_y = []   
        map_x_store = []
        map_y_store = []   
        """Generate data at 2Hz"""        
        while self.running:
            
            if self.loopcounter == 0:
                start_time = time.time()  # start time
                measured_pose_stamp_prev = 0                
                measured_pose_init = True
                lidar_timestamp_s_prev = None
                path_first_counter = 0
                            
            current_time = time.time() - start_time # current time             

            if self.time_to_run > 0 and current_time > self.time_to_run:
                self.running = False
                break
                
            # Estimated pose #
            if self.Laptop.est_pose_northings_m is not None:
                est_pose_northings_m = self.Laptop.est_pose_northings_m
                est_pose_eastings_m = self.Laptop.est_pose_eastings_m
                est_pose_yaw_rad = self.Laptop.est_pose_yaw_rad
            else:
                est_pose_northings_m = None
                est_pose_eastings_m = None
                est_pose_yaw_rad = None

            # Measured pose #
            if self.Laptop.measured_pose_timestamp_s is not None:
                measured_pose_timestamp_s = self.Laptop.measured_pose_timestamp_s
                measured_pose_northings_m = self.Laptop.measured_pose_northings_m 
                measured_pose_eastings_m = self.Laptop.measured_pose_eastings_m 
                measured_pose_yaw_rad = self.Laptop.measured_pose_yaw_rad 
            else:
                measured_pose_timestamp_s = None
                measured_pose_northings_m = None
                measured_pose_eastings_m = None
                measured_pose_yaw_rad = None

            # waypoints # 
            if northings_path != self.Laptop.northings_path or eastings_path != self.Laptop.eastings_path:                  
                self.waypoints.x.clear()
                self.waypoints.y.clear()              
                northings_path = copy.deepcopy(self.Laptop.northings_path)
                eastings_path = copy.deepcopy(self.Laptop.eastings_path)                                

            # wheel rate commands and actual #
            cmd_wheelrate_right = self.Laptop.cmd_wheelrate_right
            cmd_wheelrate_left = self.Laptop.cmd_wheelrate_left
            measured_wheelrate_right = self.Laptop.measured_wheelrate_right 
            measured_wheelrate_left = self.Laptop.measured_wheelrate_left

            # LIDAR #
            lidar_data = self.Laptop.lidar_data
            lidar_timestamp_s = self.Laptop.lidar_timestamp_s

            ######## e-frame plots ############
            #waypoints
            if path_first_counter == 0 and northings_path is not None and eastings_path is not None:
                path_first_counter = 1                

            if path_first_counter and northings_path is not None and eastings_path is not None:
                for i in range(len(northings_path)):
                    self.waypoints.cb_append_data_point(northings_path[i], eastings_path[i])

            # estimated and measured positions
            if est_pose_northings_m is not None and est_pose_eastings_m is not None:
                self.est_position.cb_append_data_point(est_pose_northings_m, est_pose_eastings_m)
            if measured_pose_northings_m is not None and measured_pose_eastings_m is not None  and measured_pose_timestamp_s > measured_pose_stamp_prev:
                self.measured_position.cb_append_data_point(measured_pose_northings_m, measured_pose_eastings_m)
    
            #lidar 
            if lidar_data is not None and lidar_timestamp_s != lidar_timestamp_s_prev:
                for point in lidar_data:
                    # if np.isnan(point[0]) == False | np.isnan(point[1] == False):
                    if not np.isnan(point[0]) and not np.isnan(point[1]):
                        map_x_store.append(point[0])
                        map_y_store.append(point[1])
                        map_x.append(point[0])
                        map_y.append(point[1])                        
                        self.lidar.cb_append_data_point(point[0], point[1])                                        
                if len(map_x) > 1500:
                    self.lidar.clear()
                    ind = np.random.choice(len(map_x_store), 1000, replace=False)   
                    for i in ind:
                        self.lidar.cb_append_data_point(map_x_store[i], map_y_store[i])
                    map_x = []
                    map_y = []
                lidar_timestamp_s_prev = lidar_timestamp_s

            
            ####### wheelrate plots #########
            if cmd_wheelrate_right is not None and cmd_wheelrate_left is not None:
                self.cmd_wheelrate_right.cb_append_data_point(cmd_wheelrate_right, current_time)
                self.cmd_wheelrate_left.cb_append_data_point(cmd_wheelrate_left, current_time)
            if measured_wheelrate_right is not None and measured_wheelrate_left is not None:
                self.measured_wheelrate_right.cb_append_data_point(measured_wheelrate_right, current_time)
                self.measured_wheelrate_left.cb_append_data_point(measured_wheelrate_left, current_time)
            
            
            ######## heading plts #######
            if measured_pose_yaw_rad is not None and measured_pose_timestamp_s > measured_pose_stamp_prev:
                self.measured_heading.cb_append_data_point(np.rad2deg(measured_pose_yaw_rad), current_time)
            if est_pose_yaw_rad is not None:
                self.est_heading.cb_append_data_point(np.rad2deg(est_pose_yaw_rad), current_time)
            
            if measured_pose_timestamp_s is not None:
                measured_pose_stamp_prev= measured_pose_timestamp_s
            
            self.loopcounter += 1  
            
            self.est_time = timestamp = time.time()
            
            self.sleeplength = 0.2+((0.2*self.loopcounter) - (self.est_time - start_time))
            if self.sleeplength <=0:
                self.sleeplength = 0
            sleep(self.sleeplength)            

    def breaker(self):
        self.Laptop.stopcommand()
        

    def start_app(self, time_to_run=-1):
        """Start the application and handle threads."""
        self.running = True
        self.time_to_run = time_to_run
        print(f"Starting app with time_to_run={time_to_run}")

        # Start the update thread for the plots
        update_thread = Thread(target=self.update)
        update_thread.daemon = True  # to ensure the update thread exits when the main program does
        update_thread.start()

        # Start the Laptop run thread
        laptop_thread = Thread(target=self.Laptop.run, args=(time_to_run,))
        laptop_thread.daemon = True  # Daemonize the thread to avoid blocking the app
        laptop_thread.start()

        
        
if __name__ == '__main__':    

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode. Defaults to False",
    )

    parser.add_argument(
        "--time",
        type=float,
        default=-1,
        help="Time to run an experiment for. If negative, run forever.",
    )

    args = parser.parse_args()

    if args.simulation: 
        print('Running laptop.py in simulation')
    else: 
        print('Running laptop.py on robot')

    app = QApplication(sys.argv)
    simulation = args.simulation
    window = Window(simulation)
    window.show()
    window.start_app(time_to_run=args.time)

    sys.exit(app.exec_())
    window.running = False