# Milestone 4: Navigation and Planning
- [Objective](#Objective)
- [Introduction](#Introduction)
	- [Semi-automatic delivery (week 8)](#Semi-automatic-delivery-week-8)
	- [Improvements and extensions (week 9)](#Improvements-and-extensions-week-9)
- [Marking schemes](#Marking-schemes)

## Objective
Implement the (semi-)automatic navigation and fruit delivery module.

## Introduction
In M4 you will work on the fruit delivery module. In the arena there will be 10 ARUCO markers, 3 person models, 3 lemons, and 3 apples again. Assume that your robot has already finished mapping the arena and estimating the locations of these objects (M2 & M3). 

Your robot now needs to identify person models without any apples nearby (within a 0.5m radius) and lemons that are too close to any person model (within a 0.5m radius). It then needs to plan for which apples and lemons to move, sch that by the end:
- there is no lemon within the 0.5m radius of any person model
- there is at least one apple within the 0.5m radius of each person model
 
We assume that if an apple or a lemon is on the path of the robot's movement, it will be pushed towards the direction that the robot is traveling to.

**Note that you are not allowed to move the ARUCO marker blocks or the person models. This will result in a marking penalty.**

### Semi-automatic delivery (week 8)
You will be working on [delivery_semiauto.py](delivery_semiauto.py) - this provides a starting point for implementing the fruit delivery module.

Running this script requires the [utility scripts](../Week01-02/util) and the [calibration parameters](../Week03-05/calibration/param/). Please copy the util and calibration folders into your working directory.

**For the simplicity of M4, you are allowed to use the true map of an arena as the input to [delivery_semiauto.py](delivery_semiauto.py). However, for M5 and the final demo, your fruit delivery module must use the estimated map produced by your SLAM and object recognition modules. In addition, you cannot teleport the robot or objects to a location or get the robot's or an object's true pose from ROS during the navigation and delivery process.** 

Complete [delivery_semiauto.py](delivery_semiauto.py) to finish the semi-automatic delivery module:

1. Calculate time the robot needs to turn towards a waypoint at [line 88](delivery_semiauto.py#L88)
2. Calculate time the robot needs to drive straight towards a waypoint at [line 93](delivery_semiauto.py#L93)
3. Calculate updated pose of the robot after it reaches a waypoint at [line 98](delivery_semiauto.py#L98)

For semi-automatic delivery, you can only provide waypoints to your robot. The robot has to figure out for itself how to get to the specified waypoints. You are not allowed to provide drive signals, such as left/right turn or stop (i.e., teleoperate).

### Improvements and extensions (week 9)
Below are suggestions that may help you improve your fruit delivery module:

1. Correct the robot's pose after every waypoint navigation step with information from the SLAM module
2. Use GUI for specifying the waypoints on a map visualisation instead of manually entering the coordinates of the waypoints
3. Create a fully automatic delivery solution which generates the waypoints automatically based on the locations of objects

You may implement your own fruit delivery script instead of using [delivery_semiauto.py](delivery_semiauto.py).

To evaluate the performance of your delivery module, copy [RetrievePose.py](RetrievePose.py) into your "catkin_ws/src/penguinpi_gazebo" directory. Once the delivery process is finished, run ```rosrun penguinpi_gazebo RetrievePose.py```, this will save the current pose of objects in Gazebo into "layout_after_delivery.txt". Note: if you encounter an error that says RetrievePose.py is not a file or not an executable, go to the src folder and type ```chmod +x RetrievePose.py```. Now run [delivery_eval.py](delivery_eval.py), which compares "layout_after_delivery.txt" against the true map, and prints person without apples nearby (delivery failure), person with lemon nearby (delivery failure), person models that seemed to have moved (collision penalty), and marker blocks that seemed to have moved (collision penalty).

**When submitting your implementation, please delete [RetrievePose.py](RetrievePose.py) as it calls for true poses of objects from Gazebo. It is only for evaluation purpose and you are not allowed to make use of it during the delivery process.**

## Marking schemes
- Live-demo evaluation (60pts): following the [same procedure](../Week03-05/M2_livemarking_runsheet_student.md) as M2's live demo marking, submit your implementation before the lab session of week 10, and then during week 10 you will be demonstrating your fruit delivery module in a new marking map. You can use the true map as the input of your delivery module for M4 marking. You will have 5 minutes to perform the delivery.
	
	Once the delivery is done, for each person model:
	- if there is at least one apple within its 0.5m radius you get 10pts
	- if there is no lemons within its 0.5m radius you get 10pts
	These marks total a possible 20x3 = 60pts.
	
	Additionally, during the delivery process:
	- every person model the robot collides with gives a penalty of -5pts
	- every ARUCO marker block the robot collides with gives a penalty of -2pts
	 
	Note: the minimum live-demo score is 0pt.
	
	For example, if during delivery the robot collided with one person model (-5pts) and two ARUCO marker blocks (2x-2pts), and after the delivery is done person_0 has 2 apples and 1 lemon within 0.5m (+10pts for apples), person_1 has no apples or lemons within 0.5m (+10pts for no lemons), person_2 has 1 apple and no lemons within 0.5m (+10pts for apple, +10pts for no lemons), your live-demo mark will be 10+10+20-5-2x2=31pts.
- Function extension (40pts):
	- If you enabled visual-based waypoint selection and demonstrated it during the live-demo evaluation you will receive 10pts.
	- If you implemented fully-automatic delivery and demonstrated it during the live-demo evaluation you will receive 30pts.
