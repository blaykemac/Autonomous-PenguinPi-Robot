# detect ARUCO markers and estimate their positions
import numpy as np
import cv2
import os, sys

sys.path.insert(0, "{}/util".format(os.getcwd()))
import util.measure as measure

class aruco_detector:
    def __init__(self, robot, marker_length=0.07):
        self.camera_matrix = robot.camera_matrix
        self.distortion_params = robot.camera_dist

        self.marker_length = marker_length
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    
    def detect_marker_positions(self, img):
        # Perform detection
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.aruco_params)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.distortion_params)

        if ids is None:
            return [], img

        # Compute the marker positions
        measurements = []
        seen_ids = []
        for i in range(len(ids)):
            idi = ids[i,0]
            # Some markers appear multiple times but should only be handled once.
            if idi in seen_ids or idi not in list(range(1, 11)):
                continue
            else:
                seen_ids.append(idi)
            
            # store list of which lm faces should be used for estimate
            valid_faces = np.zeros( len(ids), dtype = np.bool)
            
            # distance horizontally in pixels that is required for lm face to be considered valid
            permissible_width = 15
                        
            for face_index, face in enumerate(corners):
                if idi == ids[face_index, 0]:
                    top_left_x = face[0, 0, 0]
                    top_right_x = face[0, 1, 0]
                    
                    # compute horizontal distance between top corners of landmark, in pixels
                    x_width = abs(top_left_x - top_right_x)
                    
                    # only valid if we see enough pixels of lm
                    if x_width >= permissible_width:
                        valid_faces[face_index] = True
                    
            valid_faces = np.array([valid_faces]).T
            
            # if no faces are valid in view, return early as no measurements to take
            if (valid_faces == False).all():
                return [], img

            # filter out the invalid transformations
            lm_tvecs = tvecs[valid_faces].T
            lm_rvecs = rvecs[valid_faces]
            
            # compute rotation matrix R from rvec
            lm_rmats = [cv2.Rodrigues(rvec)[0] for rvec in lm_rvecs]
            
            # offset marker centre to cube centre
            cube_centre = np.array([0, 0, -0.04])
            
            # compute R * cube_centre
            rotated_centres = np.array([lm_rmat @ cube_centre for lm_rmat in lm_rmats])
            
            # compute R*cube_centre + t
            robot_offsets = rotated_centres + lm_tvecs.T
            
            #offset from camera coordinates to account for camera being 8cm in front of robot
            est_centres_cc = np.array([[r[2] + 0.08, -r[0]] for r in robot_offsets])
            
            # average the lm estimate if multiple faces detected
            mean_centre = np.mean(est_centres_cc, axis=0).reshape(-1, 1)
            adjusted_mean_centre = mean_centre
            
            # return estimate for given lm
            lm_measurement = measure.Marker(adjusted_mean_centre, idi)
            measurements.append(lm_measurement)
            
        
        # Draw markers on image copy
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

        return measurements, img_marked
