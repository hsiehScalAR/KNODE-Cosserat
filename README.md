# KNODE-Cosserat
This repository contains the hardware and code information used for the work **Knowledge-based Neural Ordinary Differential Equations for Cosserat Rod-based Soft Robots**
![alt text](https://github.com/TomJZ/KNODE-Cosserat/blob/main/CAD/full_robot_assem_isometric.PNG)
## Hardware
In ```\CAD``` we provide the SolidWorks CAD files for the full tendon-driven robot. It additionally contains the full robot assembly in a .step file, and two images of the full assembly (One picture shown above).

## Firmware
In ```\firmware``` we provide the Arduino code for the tendon-driven robot in C++. This code implements a PID controller for each tendon to track a setpoint tension. This code also reads tension from each of the custom load cells.

## ROS
In ```\ros_ws``` we provide the code for a ROS node that can control the robot using a joystick in Python. The ROS node subscribes to the joystick and maps the commands to the tensions in each tendon. The commands are sent to the robot through a serial port. This code can be modified to send custom commands to the robot instead of reading from the joystick.

## Training



## Citation
If you find any part of this repository useful and/or use it in your research, please the following publications:

    @article{knode-cosserat,
      title={Knowledge-based Neural Ordinary Differential Equations for Cosserat Rod-based Soft Robots},
      author={Jiahao, Tom Z. and Adolf, Ryan and Sung, Cynthia and Hsieh, M. Ani},
      journal={arxiv preprint},
      year={2024}}
