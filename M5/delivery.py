#!/usr/local/bin python3

# import modules
import operate
import argparse
import pygame

# main entry point
if __name__ == "__main__":

    # setup parsers to read command line arguments
    parser = argparse.ArgumentParser("Fruit delivery")
    parser.add_argument("-m", "--map", type=str, default='/home/blayke/catkin_ws/src/penguinpi_gazebo/TRUEMAP.txt') # specify the truemap directory
    parser.add_argument("-t", "--truemap", action='store_true') # specify whether to use the true map (dont think you can in final milestone or demo)
    parser.add_argument("--ip", metavar='', type=str, default='localhost') # specifies socket ip
    parser.add_argument("-p", "--port", metavar='', type=int, default=40000) # specifies socket port
    parser.add_argument("--calib_dir", type=str, default="calibration/param/") # specifies directory to parameters
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth') # specifies nn model path
    parser.add_argument("--nogui", action='store_true') # disables semiauto GUI
    parser.add_argument("--auto", action='store_true') # enable full auto delivery
    args, _ = parser.parse_known_args()

    # Initialise the main robot controller
    operator = operate.Operate(args)
    
    """ Uncomment this if you want to force a button press before starting GUI
    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2
    """
    try:
        # main control loop where we run all necessary functions in class Operate()
        while True:
            # check for any events such as keyboard or mouse presses
            operator.update_input()
            
            # run full auto waypoint creator
            operator.automate_waypoint()
            
            # take image from camera
            operator.take_pic()
            
            # navigate to waypoint
            operator.navigate_to_waypoint()
            
            # generate drive signal from the 
            drive_meas = operator.control()
            
            # run predict and update step of SLAM given the drive signal generated previously
            operator.update_slam(drive_meas)
            
            # save slam map, detected objects and raw camera image
            operator.record_data()
            operator.save_image()
            
            # perform object segmentation
            operator.detect_target()
            
            # visualise
            operator.draw()
            pygame.display.update()
            
            # print state for debugging
            print(f"ekf state: {operator.ekf.robot.state}")
            
    except Exception as exc:
        # stop robot from moving if script crashes or we close program
        operator.pibot.set_velocity([0, 0])
        print(exc)
        
        pass