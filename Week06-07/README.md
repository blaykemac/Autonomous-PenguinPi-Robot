# Milestone 3: Object Recognition and Localisation
- [Objectives](#Objectives)
- [Introduction](#Introduction)
    - [An index of provided resources](#An-index-of-provided-resources)
- [Data collection](#Data-collection)
- [Training your neural network](#Training-your-neural-network)
- [Estimating target poses](#Estimating-target-poses)
- [Marking schemes](#Marking-schemes)

## Objectives
1. Week 6: collect data and train your target detector with the helper scripts
2. Week 7: complete the codes to estimate pose of detected targets

## Introduction
**Note: make sure you have saved your previous development elsewhere before pulling the repo.**

### An index of provided resources
- [operate.py](operate.py) is the central control program that combines keyboard teleoperation (M1), SLAM (M2), and object recognitions (M3). It requires the [utility scripts](../Week01-02/util), the [GUI pics](../Week01-02/pics) from M1, and the [calibration parameters](../Week03-05/calibration/param/) and [SLAM scripts](../Week03-05/slam/) of M2. In addition, it calls for the [detector model](#Training-your-neural-network) that you will develop in M3.
- [ECE4078_brick.world](ECE4078_brick.world) and [ECE4078_brick.launch](ECE4078_brick.launch) are used for launching a new testing arena with wall and floor textures added.
- [data_collector.zip](https://drive.google.com/file/d/1pCArWQMGu7AeYi8JU107iHEzn2mpcXZ5/view?usp=sharing) contains scripts required for generating the [synthetic dataset](#Data-collection) used for training your target detector.
- [network](network/) contains [scripts](network/scripts) required for [training a target detector model](#Training-your-neural-network) locally and for training on [Google Colab](network/ECE4078_2021_Lab_M3_Colab).
- [CV_eval.py](CV_eval.py) is used for [evaluating the pose estimations](#Estimating-target-poses) given by your detector compared to the groundtruth map. The pose estimation is done by [TargetPoseEst.py](TargetPoseEst.py) which you will complete.

## Data collection
### Environment variances
Download [ECE4078_brick.world](ECE4078_brick.world) and [ECE4078_brick.launch](ECE4078_brick.launch) to your "worlds" and "launch" folder inside "catkin_ws/src/penguinpi_gazebo". Now open a terminal and run: 
```
source ~/catkin_ws/devel/setup.bash
roslaunch penguinpi_gazebo ECE4078_brick.launch
``` 
You should see an areana with brick walls and wooden floors. You can then spawn objects inside this new environment by opening a new terminal and run 
```
source ~/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py
```
Try changing the wall and floor materials during development to test the robustness of your detector. To do so, open [ECE4078_brick.world](ECE4078_brick.world) in a text editor, search and replace "Gazebo/Bricks" (4 occurrences in total) with other pre-defined materials listed [here](http://wiki.ros.org/simulator_gazebo/Tutorials/ListOfMaterials) to change the wall material, and search and replace "Gazebo/Wood" (1 occurrence in total) with other pre-defined materials to change the floor material.

![Brick Walls](Screenshots/BrickWallWorld.png?raw=true "Brick Walls")

### Generating synthetic data
As training a neural network requires a lot of data, but manual data collection is time consuming, a ROS-based data collector is provided for you to generate synthetic data for training. Download [data_collector.zip](https://drive.google.com/file/d/1pCArWQMGu7AeYi8JU107iHEzn2mpcXZ5/view?usp=sharing) and unzip it into your "catkin_ws/src/" folder.

Open a terminal and install required packages for **python 2** used by the ROS data collector:
```
sudo apt install python-pip
python -m pip install tqdm h5py
```
In catkin_ws/src/data_collector/data_collector.py, replace the camera matrix at Line 33 with your [camera parameters](../Week03-05/calibration/param/intrinsic.txt) computed from the camera calibration process in M2.

Close Gazebo if it is running. Then open the Gazebo photo studio (an empty world with lighting set-up) by running the following command in a terminal 
```
source ~/catkin_ws/devel/setup.bash
roslaunch data_collector photo_studio.launch gui:=true
```

Now open a new terminal and generate the synthetic dataset by typing the commands below (note that we are using "python" instead of "python3")
```
source ~/catkin_ws/devel/setup.bash
roscd data_collector
python data_collector.py --dataset_name Synth --sample_per_class 1000
```
This creates a folder in "catkin_ws/src/data_collector/dataset/" called "Synth", in which there is a folder for each target class containing 1000 synthetic training images generated with grey background in the images folder (if you check the Gazebo window while the data is being generated, you can see the models appearing at random locations for taking each of these images). The "images" folder saves the original image, while the "labels" folder saves the segmentation labels (silhouette of the target model) with small variation in the black level for training your neural network. The black level variation is amplified in "labels_readable" so that you can visualize what the labels look like if you want.

![Person model with grey background](Screenshots/person_grey.jpg?raw=true "Person model with grey background")

Now we need to replace the grey background with random background images to increase the detector's robustness. Open a terminal and run the following commands (note that we are using "python", not "python3")
```
cd ~/catkin_ws/src/data_collector/
python randomise_background.py --dataset_name Synth
``` 
You should now see the images in "catkin_ws/src/data_collector/dataset/Synth" with random background pictures added to them. The background pictures are stored in "catkin_ws/src/data_collector/textures/random" and you can remove part of it or add your own collection.

![Person model with random background](Screenshots/person_rand.jpg?raw=true "Person model with random background")

In the same terminal, after the background shuffling is done, run the dataset split script (note that we are using "python", not "python3"):
```
python split_dataset.py --sim_dataset Synth --training_ratio 0.8
``` 
This will separate the synthetic dataset randomly into a training set containing 80% of all images and an evaluation set containing 20% of all images, specified by "train.hdf5" and "eval.hdf5" generated under "catkin_ws/src/data_collector/dataset/", which will be used for training your neural network.

## Training your neural network
Install [PyTorch](https://pytorch.org/) (select "Pip" as Package and "None" for CUDA) and other dependencies for **python 3** using the following commands:
```
python3 -m pip install pandas h5py tqdm
python3 -m pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install -U PyYAML
```

We use [ResNet-18](https://pytorch.org/hub/pytorch_vision_resnet/) as our model, which is a deep convolutional neural network pre-trained on the ImageNet dataset. The structure of the ResNet18 model is specified in [res18_skip.py](network/scripts/res18_skip.py).

![ResNet-18](Screenshots/ResNet.jpg?raw=true "ResNet-18")

To train the neural network with the data you generated, run the commands below in a terminal (note that we are using "python3" now). Make sure you edit the command to point to your correct directory.
```
cd ~/YourDir/Week06-07/network/scripts/
python3 main.py --dataset_dir ~/catkin_ws/src/data_collector/dataset --model_dir model
```
You can change the [default parameters](network/scripts/args.py) by adding flags, such as ```--lr 0.05```, when running [main.py](network/scripts/main.py). Parameters that you can change, what they represent and their default values are specified in [args.py](network/scripts/args.py).

You will see some information printed regarding how a model performs for training and evalution during each epoch. Once training is done, a best performing model will be saved as "~/YourDir/Week06-07/network/scripts/model/model.best.pth". This model should be able to detect the different targets (apple, lemon, person) and segment them from the background. Delete old checkpoints (.pth) before training new models.

If you would like to train your neural network on Google Colab, upload the [ECE4078_2021_Lab_M3_Colab folder](network/ECE4078_2021_Lab_M3_Colab) with your dataset included (images, labels, and the hdf5 files) to your Google Drive. Open [main_colab.ipynb](network/ECE4078_2021_Lab_M3_Colab/main_colab.ipynb) in Colab and then run its cells (you can change the default parameters in [args.py](network/ECE4078_2021_Lab_M3_Colab/args.py)). If you would like to use GPU or TPU with Colab, in the top menu, go to "Runtime" -> "Change runtime type" -> "Hardware accelerator" -> select "GPU" or "TPU" in the dropdown menu. After the training is done, you can view the performance of the trained network on 4 test images using [detector_debugger.ipynb](network/ECE4078_2021_Lab_M3_Colab/detector_debugger.ipynb), and you can download the generated best model "model.best.pth" to your local directory for the detector to use.

![Segmenting target from background](Screenshots/Prediction.jpg?raw=true "Segmenting target from background")

Once you are happy with the model's performance, you can use it for your robot. Load the [Gazebo world](#Environment-variances) with environment textures and targets, and then in a terminal run 
```
cd ~/YourDir/Week06-07/
python3 operate.py
```
You may need to change the path to the best performing model by running [operate.py](operate.py) with the ```--ckpt``` flag. Once the GUI is launched, you can run the target detector by pressing "p", this shows the detector's output in the bottom left of the GUI. 

![GUI view of the detector](Screenshots/DetectorWorking.png?raw=true "GUI view of the detector")

## Estimating target poses
To estimate pose of targets, you will need to run the SLAM component (press ENTER to run) while running the target detector, so that the robot's pose is known. Every time you want to perform object detection, press "p" to run the detector, then press "n" to save the robot's current pose estimated by SLAM (as "/YourDir/Week05-06/lab_output/image.txt") as well as the corresponding detector's segmentation labels (similar to the images in the "labels" folder when generating the synthesized data, these segmentation labels appear all black. You can press "i" to save the raw image in addition for visual debugging of the pose estimation). After you have collected the detector's outputs for all targets, you can press "s" to save SLAM map and then exit the GUI by pressing ESC.

**Complete [TargetPoseEst.py](TargetPoseEst.py)** to estimate the locations of the apples, lemons, and person models based on the detector's outputs and the robot's poses.

- Replace [Lines 81-86](TargetPoseEst.py#L81) with your own codes to compute the target pose using the robot's pose and the detector's output. There are 3 apples, 3 lemons, and 3 person models in the marking map, so you can only output at most 3 estimations per target type in the estimation output.
Replace [Lines 106-128](TargetPoseEst.py#L106) with your own codes to merge the estimations with a better way than taking the first 3 estimations. The [TargetPoseEst.py](TargetPoseEst.py) generates an estimation result file as "/YourDir/Week06-07/lab_output/targets.txt", which is in the same format as the groundtruth maps.

You can use [CV_eval.py](CV_eval.py) to evaluate performance of your target pose estimation. Run 
```
python3 CV_eval.py TRUEMAP.txt lab_output/targets.txt
```
This computes the Euclidean distance between each target and its closest estimation (the estimation error) and returns the average estimation error over all targets. If more than 3 estimations are given for a target type, the first 3 estimations will be used in the evaluation.

## Marking schemes
- Please submit your completed "TargetPoseEst.py" and your trained neural network as "model.best.pth" (don't submit your training dataset).
- Your trained model will be applied to a new marking map with a set of images taken at various points in the marking map for your detector to generate the predictions. The groundtruth robot poses associated with each image will be used for marking so the estimation performance will only depend on your neural network's performance and your pose estimation script. The resulting target pose estimation will be compared against the groundtruth map using [CV_eval.py](CV_eval.py) for simulator-based evaluation.
- For simulator-based evaluation (80pts), the average estimation error calculated by [CV_eval.py](CV_eval.py) will be used for marking. For live-demo evaluation (20pts), you have 5 minutes to drive your robot in a marking map. Demonstrating that your robot can recognize at least one apple, one lemon, and one person is 5pts each (in the GUI the detector shows a reasonable object segmentation). Demonstrating that your robot can generate pose estimations of at least one target is 5pts. Your M3 score is calculated as (1-estimation_error) x 80 + live_demo_score (if your estimation error is bigger than 1 then your simulator-based evaluation score would be 0). For example, if your estimation error is 0.4 in the simulator-based evaluation, and your robot recognizes and generates pose estimations for an apple and a lemon in the live demo, your M3 score will be (1-0.4)x80+15=63.
