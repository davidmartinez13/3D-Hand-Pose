import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
from matplotlib import pyplot as plt
import sys, time, math

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def draw_hand_pose(image, camera_matrix,camera_distortion, marker_size, rvec,tvec, z_rot=-1):
        world_points = np.array([
            3.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, -3.0 * z_rot
        ]).reshape(-1, 1, 3)

        img_points, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, camera_distortion)
        img_points = np.round(img_points).astype(int)
        img_points = [tuple(pt) for pt in img_points.reshape(-1, 2)]

        cv2.line(image, img_points[0], img_points[1], (0,0,255), 2)
        cv2.line(image, img_points[1], img_points[2], (0,255,0), 2)
        cv2.line(image, img_points[1], img_points[3], (255,0,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'X', img_points[0], font, 0.5, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Y', img_points[2], font, 0.5, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(image, 'Z', img_points[3], font, 0.5, (255,0,0), 2, cv2.LINE_AA)
        # Position of hand respect to camera , if camera is steady and hand moving
        str_position = "x=%4.0f y=%4.0f z=%4.0f"%(tvec[0], tvec[1], tvec[2])
        cv2.putText(image, str_position,(img_points[2][0]+10,img_points[2][1]-30), 
                        font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Orientation of hand respect to camera , if camera is steady and hand moving
        # Rot matrix hand to camera:
        R_cam2hand = np.matrix(cv2.Rodrigues(rvec)[0])
        R_hand2cam = R_cam2hand.T
        # Rotation euler angles hand to camera (flipped first)
        x_rot_hand, y_rot_hand, z_rot_hand = rotationMatrixToEulerAngles(R_flip*R_hand2cam)
        str_rot_hand = "x_rot=%4.0f y_rot=%4.0f z_rot=%4.0f"%(
                            math.degrees(x_rot_hand),
                            math.degrees(y_rot_hand),
                            math.degrees(z_rot_hand))
        cv2.putText(image, str_rot_hand, (img_points[2][0]+10,img_points[2][1]-15), 
                    font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Position of camera respect to hand, if hand is steady and camera moving
        pos_camera = -R_hand2cam * tvec
        str_pos_cam = "Camera Transl x=%4.0f y=%4.0f z=%4.0f"%(pos_camera[0], pos_camera[1], pos_camera[2])
        cv2.putText(image, str_pos_cam, (0, 200), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Orientation of camera respect to hand, if hand is steady and camera moving
        x_rot_cam, y_rot_cam, z_rot_cam = rotationMatrixToEulerAngles(R_flip*R_hand2cam)
        str_rot_cam = "Camera Rot x_rot=%4.0f y_rot=%4.0f z_rot=%4.0f"%(
                            math.degrees(x_rot_cam),
                            math.degrees(y_rot_cam),
                            math.degrees(z_rot_cam))
        cv2.putText(image, str_rot_cam, (0, 250), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        
def compute_hand_pose(image, results, hand_points, hand_plane,camera_matrix,camera_distortion,draw_points=False):
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        joint_img_pts = np.zeros((len(hand_points),2))
        joint_world_pts = np.zeros((len(hand_points),3))
        #Loop through joint sets
        for i, joint in enumerate(hand_points):
            joint_img_pt = np.array([hand.landmark[joint].x, hand.landmark[joint].y])
            joint_img_pt = np.multiply(joint_img_pt, [width, height]).astype(int)
            joint_world_pt = np.array(hand_plane[str(joint)])
            joint_img_pts[i] = joint_img_pt
            joint_world_pts[i] = joint_world_pt
            if draw_points:
                cv2.putText(image, str(tuple(joint_img_pt)), tuple(joint_img_pt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        _,rvec, tvec = cv2.solvePnP(joint_world_pts, joint_img_pts, camera_matrix, camera_distortion)
        draw_hand_pose(image, camera_matrix, camera_distortion, 8, rvec, tvec)
    return image

def draw_finger_angles(image, results, joint_list):
    
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
                
            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [width, height]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, 
                            hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [width,height]).astype(int))
            
            output = text, coords
            
    return output
hand_plane = {'0':[0.0, 0.0, 0.0], #wrist
            '17':[7.0, 0.0, 0.0], #pinky_base
            '5':[6.0, 5.5, 0.0], #index_base
            '2':[2.0, 6.5, 0.0], #thumb_base
            '9':[7.5, 3.7, 0.0], #middle_base
            '13':[7.7, 2.0, 0.0] #ring_base
}
# 180 deg rotation matrix around the x axis    
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

calib_path = "hand_gesture_depth/"
camera_matrix = np.loadtxt(calib_path+'camera_matrix.txt', delimiter=',')
camera_distortion = np.loadtxt(calib_path+'distortion.txt', delimiter=',')    

joint_list = [[8,7,6], [12,11,10], [16,15,14], [20,19,18]]
hand_points = [0, 17, 5, 2, 9, 13]
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_hands.HandLandmark.WRIST
width = 1280
height = 720
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        # Set flag
        image.flags.writeable = False
        # Detections
        results = hands.process(image)
        # Set flag to true
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image, hand, mp_hands.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
                # left or right hand
                # if get_label(num, hand, results):
                #     text, coord = get_label(num, hand, results)
                #     cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                # (255, 255, 255), 2, cv2.LINE_AA)
            # Draw angles to image from joint list
            # draw_finger_angles(image, results, joint_list)
            compute_hand_pose(image, results, hand_points, hand_plane, camera_matrix, camera_distortion)

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) == 27:
            break

cap.release()
cv2.destroyAllWindows()