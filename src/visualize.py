import cv2
import numpy as np
import math
#2021/11/03, If you use WSL in windows, you need to change environ accordingly, 
# Please refer to the README.md file for detail.
class Visualizer:
    def __init__(self, frame, fd_img, fd_coords, ld_coords, eye_l_image, eye_r_image, hpe_output,mouse_xy,gaze_val):
        self.frame = frame
        self.fd_img = fd_img
        self.fd_coords = fd_coords
        self.ld_coords = ld_coords
        self.hpe_output = hpe_output
        self.mouse_xy = mouse_xy
        self.eye_l_image = eye_l_image
        self.eye_r_image = eye_r_image
        self.gaze_val = gaze_val

    def visualizer(self,visualization_flags):
        if len(visualization_flags) != 0:
            vis_frame = self.fd_img.copy()         
            if 'fd' in visualization_flags:
                cv2.rectangle(self.frame, (self.fd_coords[0], self.fd_coords[1]),
                        (self.fd_coords[2], self.fd_coords[3]), (0, 255, 0),5)        
            if 'hpe' in visualization_flags:
                # returns frame following the original input frame
                cv2.putText(self.frame,
                            "yaw:{:.1f} | pitch:{:.1f} | roll:{:.1f}".format(self.hpe_output[0], self.hpe_output[1], self.hpe_output[2]),
                            (60, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)                                   
            if 'ld' in visualization_flags:
                vis_frame = cv2.rectangle(vis_frame, (self.ld_coords[0]), (self.ld_coords[1]), (255, 0, 255))
                vis_frame = cv2.rectangle(vis_frame, (self.ld_coords[2]), (self.ld_coords[3]), (255, 0, 255))            
            if 'ge' in visualization_flags:
                gaze_vec_norm = self.gaze_val / np.linalg.norm(self.gaze_val)                            # normalize the gaze vector
                roll = self.hpe_output[2]
                vcos = math.cos(math.radians(roll))
                vsin = math.sin(math.radians(roll))
                tmpx =  gaze_vec_norm[0]*vcos + gaze_vec_norm[1]*vsin
                tmpy = -gaze_vec_norm[0]*vsin + gaze_vec_norm[1]*vcos
                gaze_vec_norm = [tmpx, tmpy]
                # Store gaze line coordinations
                gaze_lines = []
                for i in [0,2]:
                    #left top
                    coord1 = (self.ld_coords[i][0],                                 self.ld_coords[i][1])
                    #left top + gaze vector
                    coord2 = (self.ld_coords[i][0]+int((gaze_vec_norm[0]+0.)*100), self.ld_coords[i][1]-int((gaze_vec_norm[1]+0.)*100))
                    gaze_lines.append([coord1, coord2])  
                vis_frame = cv2.line(vis_frame, gaze_lines[0][0], gaze_lines[0][1], (0, 0, 255), 4)
                vis_frame = cv2.line(vis_frame, gaze_lines[1][0], gaze_lines[1][1], (0, 0, 255), 4)
            # stack the images horizontally to see face zoom up image
            vis_frame = np.hstack((cv2.resize(self.frame, (500, 500)), cv2.resize(vis_frame, (500, 500))))
            cv2.namedWindow("Visualizer", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Visualizer", vis_frame)
        else:
            vis_frame = cv2.resize(self.frame,(500,500))
            cv2.imshow("Visualizer", vis_frame)
        return vis_frame            
