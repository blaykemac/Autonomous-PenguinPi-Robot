# Milestone 1: Teleoperating the Robot with your Keyboard

We will be using the [PenguinPi robot](https://cirrusrobotics.com.au/products/penguinpi/) in the [Gazebo simulator](http://gazebosim.org/) for our lab project.

This first milestone will allow you to become more familar with the simulator environment, and with some useful controls.

## Objective 1: Setting up your Environment
Team members(s) attending the lab session in person should test using the physical robot, while those attending the lab sessions remotely will need to set up the simulator enviroment. Whether working in person or remotely, all code development and testing should be done within the virtual machine (VM) provided.

1. The [Installation Guide](InstallationGuide.md) provides detailed instructions on installing the VM, and then setting up and testing the simulator environment, or connecting to the physical robot.
2. Within your VM, clone this repo by typing ```git clone https://github.com/tianleimin/ECE4078_Lab_2021``` inside your terminal.
3. Navigate to this week's lab by typing the commmands ``` cd ECE4078_Lab_2021/Week01-02/```

## Objective 2: Implement Keyboard Teleoperation

You will implement keyboard teleoperations by editing [line 137 - 150 of operate.py](operate.py#L137).

You will also need [the pics folder for painting the GUI](pics/), and [the util folder for utility scripts](util/), already contained within the repo.

- **Using Simulator Environment**
  - You can run [operate.py](operate.py) by typing ```python3 operate.py``` in a terminal. You will need to launch the simulator environment first.
This command will open a GUI with the robot's camera view shown.
- **Using Physical Robot**
  - If you are operating a physical robot, you will need to include flag for its IP address and port in your command (see [Installation Guide](InstallationGuide.md) for details).

**You don't have to use the provided scripts.** Feel free to be creative and write your own scripts for teleoperating the robot with keyboard.

## Marking Guidelines
- Basic implementation submitted on Moodle (80pt):
  - Drive foward +20pt
  - Drive backward +20pt
  - Turn left +20pt
  - Turn right +20pt
- Demonstration (20pt): demonstrate your teleoperation in the simulator or on the physical robot to the demonstrators during Week 3 lab session.

