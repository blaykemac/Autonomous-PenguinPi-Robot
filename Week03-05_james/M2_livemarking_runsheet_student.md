# Running sheet for M2 live marking

## Preparation
1. Each live marking session should take 10-15 minutes
2. In the breakout room, tell your demonstrator who is the one running the demo (the driver), then the driver will share their screen. The screen-sharing will be maintained throughout the live marking session
3. Please make sure only one display is on for the driver during the live demo:
	- Windows: Settings -> System -> Display
	- Linux: ```xrandr | grep connected | wc -l``` (type this in terminal and then count the lines returned)
	- Mac: System Preferences -> Displays
4. Download your M2 implementation submission from Moodle
5. Unzip the submission on desktop, then open the unzipped folder, open the "lab_output" folder and delete "slam.txt" if exists
6. Your demonstrator will provide a new marking map for you. Open an empty text editor, copy the map in there, then save it as "catkin_ws/src/penguinpi_gazebo/map2.txt". You may want to [enable Shared clipboard](https://askubuntu.com/questions/1223771/virtualbox-6-1-4-shared-clipboard-does-not-work-ubuntu-18-04) to allow copy-pasting between host and VM before the live demo, or you can use a USB stick as the intermediate data transfer, or do it through the internet, like sending yourself a message / email, or joining Zoom from inside the VM.
7. Launch the empty world in gazebo (open a terminal and type the following command)
	```
	source ~/catkin_ws/devel/setup.bash
	roslaunch penguinpi_gazebo ECE4078.launch
	```
8. Spawn the objects using **map2.txt** (open a terminal and type the following command)
	```
	source ~/catkin_ws/devel/setup.bash
	rosrun penguinpi_gazebo scene_manager.py -l map2.txt
	```

## Live SLAM
9. You have 1 minute to discuss among the group how you would like to drive the robot in this new arena
10. Go into the downloaded submission folder and start the slam by running "operate.py"
	```
	cd ~/Desktop/Lab*_M2_Group*/
	python3 operate.py
	```
11. Remember to press ENTER to start the SLAM. You will have 5 minutes (countdown shown on GUI) to perform SLAM. Other group members can make suggestions to the driver during SLAM too. When you are done or when the timer runs out, save the generated map by pressing "s", and then exit the program by pressing ESC. You only have one try.
13. Rename the generated map "/Desktop/Lab*_M2_Group*/lab_output/slam.txt" as "slam_labsession_groupnumber.txt" (e.g., "slam_1_01.txt") and submit it on Moodle's Lab Project page in "M2: Map (lab session *)"
14. Your submitted implementation will be checked (calling for gazebo true pose will result in your M2 being marked 0)
15. Your submitted slam map will be marked using "slam_eval.py". M2 mark is computed as (0.1-Aligned_RMSE)*800 + NumberOfFoundMarkers*2
