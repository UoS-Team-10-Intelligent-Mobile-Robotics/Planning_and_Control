# Planning_and_Control

This is the repository for the first series of Practical sessions (P1 to P3) of the Intelligent Mobile Robotics module. The repository is designed around the build from the module. 

## Directory tree description

  src -- main code files are stored here, we'll be editing only show_laptop.py and laptop.py \
  log -- data logs are stored here as JSON files. Need to be renamed after every run of the code, otherwise it only saves it as a timestamp \
  webots -- necessary files for the webots simulator, like world, robot etc. \

## JSON file structure

  The JSON files closely resemble the hierarchy given by the ZeroROS. Objects are described by different topics and each topic has different attributes. The topics are updated by subscribers and publishers, which are essentially different parts of the robot (LiDAR, motors, encoders etc.)
