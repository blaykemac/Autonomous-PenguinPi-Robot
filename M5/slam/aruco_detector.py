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
            
        #print(f"ids: {ids}")

        if ids is None:
            return [], img

        # Compute the marker positions
        measurements = []
        seen_ids = []
        for i in range(len(ids)):
            idi = ids[i,0]
            # Some markers appear multiple times but should only be handled once.
            if idi in seen_ids:
                continue
            else:
                seen_ids.append(idi)

            #print(f"idi: {idi} ##################################")
            
            valid_faces = np.zeros( len(ids), dtype = np.bool)
            permissible_width = 15
                        
            for face_index, face in enumerate(corners):
                if idi == ids[face_index, 0]:
                    top_left_x = face[0, 0, 0]
                    top_right_x = face[0, 1, 0]
                    
                    x_width = abs(top_left_x - top_right_x)
                    
                    if x_width >= permissible_width:
                        valid_faces[face_index] = True
                    
            valid_faces = np.array([valid_faces]).T
            if (valid_faces == False).all():
                return [], img

                       
            # our version
            lm_tvecs = tvecs[valid_faces].T
            #print(f"lm_tvecs: {lm_tvecs}")
            #print(f"tvecs: {tvecs[idi == ids].T}")
            
            
            lm_rvecs = rvecs[valid_faces]
            
            lm_rmats = [cv2.Rodrigues(vec)[0] for vec in lm_rvecs]
            # offset marker centre to cube centre
            cube_centre = np.array([0, 0, -0.04])
            rotated_offsets = np.array([lm_rmat @ cube_centre for lm_rmat in lm_rmats])
            
            robot_offsets = rotated_offsets + lm_tvecs.T
            est_centres_cc = np.array([[r[2] + 0.08, -r[0]] for r in robot_offsets])
            
            #may also need to offset from camera coordinates to account for camera being 10cm in front of robot
            mean_centre = np.mean(est_centres_cc, axis=0).reshape(-1, 1)
            adjusted_mean_centre = mean_centre
            
            lm_measurement = measure.Marker(adjusted_mean_centre, idi)
            measurements.append(lm_measurement)
            
            #print(f"lm_rvecs: {lm_rvecs}")
            #print(f"lm_rmats: {lm_rmats}")
            #print(f"rotated offsets: {rotated_offsets}")
            #print(f"robot offsets: {robot_offsets}")
            print(f"est_centres_cc: {est_centres_cc}")
            print(f"id, mean centers: {idi}, {mean_centre}")
            
        
        # Draw markers on image copy
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

        return measurements, img_marked
