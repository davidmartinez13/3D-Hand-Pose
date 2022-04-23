from opcua import ua
from opcua import Client
import cv2
import numpy as np
import time
from HandDetection import HandDetection
class OPCUA_Client:
    def __init__(self):
        self.jog_dict = {
        'jog_x_p' : False,
        'jog_x_m' : False,
        'jog_y_p' : False,
        'jog_y_m' : False,
        'jog_z_p' : False,
        'jog_z_m' : False,
        'jog_a_p' : False,
        'jog_a_m' : False,
        'jog_b_p' : False,
        'jog_b_m' : False,
        'jog_c_p' : False,
        'jog_c_m' : False
        }
        print('[INFO]:Client object started.')

    def connect_OPCUA_server(self):
        """
        Connects OPC UA Client to Server on PLC.

        """
        password = "CIIRC"
        self.client = Client("opc.tcp://user:"+str(password)+"@0.0.0.0:0000/")
        self.client.connect()
        print('[INFO]: Client connected.')

    def get_nodes(self):
        """
        Using the client.get_node method, it gets nodes from OPCUA Server on PLC.

        """
        self.Start_Prog = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."example"."pickPlace"."command"."start"')
        self.Conti_Prog = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."example"."pickPlace"."command"."continue"')
        self.Stop_Prog = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."example"."pickPlace"."command"."interrupt"')
        self.Abort_Prog = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."powerRobot"."command"."abort"')
        self.Prog_Done = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."example"."pickPlace"."status"."done"')
        self.Stop_Active = self.client.get_node(
            'ns=3;s="InstPickPlace"."instInterrupt"."BrakeActive"')
        self.Rob_Stopped = self.client.get_node(
            'ns=3;s="InstKukaControl"."instAutomaticExternal"."ROB_STOPPED"')
        self.Conveyor_Left = self.client.get_node(
            'ns=3;s="conveyor_left"')
        self.Conveyor_Right = self.client.get_node(
            'ns=3;s="conveyor_right"')
        self.Gripper_State = self.client.get_node(
            'ns=3;s="gripper_control"')
        self.Encoder_Vel = self.client.get_node(
            'ns=3;s="Encoder_1".ActualVelocity')
        self.Encoder_Pos = self.client.get_node(
            'ns=3;s="Encoder_1".ActualPosition')

        self.Act_Pos_X = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."X"')
        self.Act_Pos_Y = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."Y"')
        self.Act_Pos_Z = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."Z"')
        self.Act_Pos_A = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."A"')
        self.Act_Pos_B = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."B"')
        self.Act_Pos_C = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."C"')
        self.Act_Pos_Turn = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."Turn"')
        self.Act_Pos_Status = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."Status"')

        self.Jog_X_P = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."jog"."command"."jog01Plus"')
        self.Jog_X_M = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."jog"."command"."jog01Minus"')
        self.Jog_Y_P = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."jog"."command"."jog02Plus"')
        self.Jog_Y_M = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."jog"."command"."jog02Minus"')
        self.Jog_Z_P = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."jog"."command"."jog03Plus"')
        self.Jog_Z_M = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."jog"."command"."jog03Minus"')
        self.Jog_A_P = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."jog"."command"."jog04Plus"')
        self.Jog_A_M = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."jog"."command"."jog04Minus"')
        self.Jog_B_P = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."jog"."command"."jog05Plus"')
        self.Jog_B_M = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."jog"."command"."jog05Minus"')
        self.Jog_C_P = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."jog"."command"."jog06Plus"')
        self.Jog_C_M = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."jog"."command"."jog06Minus"')

    def get_actual_pos(self):
        """
        Reads the actual position of the robot TCP with respect to the base.
    
        Returns:
        tuple: Actual pos. of robot TCP: x, y, z, a, b, c as float. Status, turn as int.

        """
        x_pos = self.Act_Pos_X.get_value()
        y_pos = self.Act_Pos_Y.get_value()
        z_pos = self.Act_Pos_Z.get_value()
        a_pos = self.Act_Pos_A.get_value()
        b_pos = self.Act_Pos_B.get_value()
        c_pos =self.Act_Pos_C.get_value()
        status_pos = self.Act_Pos_Status.get_value()
        turn_pos = self.Act_Pos_Turn.get_value()
        x_pos = round(x_pos,2)
        y_pos = round(y_pos,2)
        z_pos = round(z_pos,2)
        a_pos = round(a_pos,2)
        b_pos = round(b_pos,2)
        c_pos = round(c_pos,2)
        return x_pos, y_pos, z_pos, a_pos, b_pos, c_pos, status_pos, turn_pos
    def stop_jog(self):
        self.Jog_X_M.set_value(ua.DataValue(False))
        self.Jog_X_P.set_value(ua.DataValue(False))
        self.Jog_Y_M.set_value(ua.DataValue(False))
        self.Jog_Y_P.set_value(ua.DataValue(False))
        self.Jog_Z_M.set_value(ua.DataValue(False))
        self.Jog_Z_P.set_value(ua.DataValue(False))

        self.Jog_A_M.set_value(ua.DataValue(False))
        self.Jog_A_P.set_value(ua.DataValue(False))
        self.Jog_B_M.set_value(ua.DataValue(False))
        self.Jog_B_P.set_value(ua.DataValue(False))
        self.Jog_C_M.set_value(ua.DataValue(False))
        self.Jog_C_P.set_value(ua.DataValue(False))
        time.sleep(0.3)

    def update_jog(self, jog_key):
        self.jog_dict = {key: not self.jog_dict[key] if key is jog_key 
                            else False for key in self.jog_dict}
        self.stop_jog()
        self.Jog_X_M.set_value(ua.DataValue(self.jog_dict['jog_x_m']))
        self.Jog_X_P.set_value(ua.DataValue(self.jog_dict['jog_x_p']))
        self.Jog_Y_M.set_value(ua.DataValue(self.jog_dict['jog_y_m']))
        self.Jog_Y_P.set_value(ua.DataValue(self.jog_dict['jog_y_p']))
        self.Jog_Z_M.set_value(ua.DataValue(self.jog_dict['jog_z_m']))
        self.Jog_Z_P.set_value(ua.DataValue(self.jog_dict['jog_z_p']))

        self.Jog_A_M.set_value(ua.DataValue(self.jog_dict['jog_a_m']))
        self.Jog_A_P.set_value(ua.DataValue(self.jog_dict['jog_a_p']))
        self.Jog_B_M.set_value(ua.DataValue(self.jog_dict['jog_b_m']))
        self.Jog_B_P.set_value(ua.DataValue(self.jog_dict['jog_b_p']))
        self.Jog_C_M.set_value(ua.DataValue(self.jog_dict['jog_c_m']))
        self.Jog_C_P.set_value(ua.DataValue(self.jog_dict['jog_c_p']))
        print(self.jog_dict)
        time.sleep(0.3)
        

rc = OPCUA_Client()
hd = HandDetection()
rc.connect_OPCUA_server()
gripper = False
width = 1280
height = 720
# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
x_hand = 0.0 
while True:
    rc.get_nodes()
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    # Set flag
    image.flags.writeable = False
    image_, canvas = hd.hand_detector(image.copy())
    print(hd.hand_pose_dict)
    if hd.hand_pose_dict:
        x_hand = hd.hand_pose_dict['x_hand']
        z_hand = hd.hand_pose_dict['z_hand']
        print(x_hand)
        if x_hand > 15.0 and x_hand < 16.0 :
            rc.update_jog('jog_y_p')
            print('x_increment')

        if x_hand < -15.0 and x_hand > -16.0:
            rc.update_jog('jog_y_m')
            print('x_decrement')

        # if z_hand > 50.0 and z_hand < 59.0 :
        #     rc.update_jog('jog_x_p')
        #     print('z_increment')

        # if z_hand < 30.0 and z_hand > 29.0:
        #     rc.update_jog('jog_x_m')
        #     print('z_decrement')
    cv2.imshow('Hand Tracking', image_)
    cv2.imshow('canvas', canvas)
    key = cv2.waitKey(1)
    if key == 27:
        rc.stop_jog()
        rc.client.disconnect()
        break
    if key == ord('g'):
        gripper = not gripper
        rc.Gripper_State.set_value(ua.DataValue(gripper))
        time.sleep(0.4)

    if key == ord('e') :
        rc.update_jog('jog_x_p')
        
    if key == ord('q') :
        rc.update_jog('jog_x_m')
        
    if key == ord('d') :
        rc.update_jog('jog_y_p')

    if key == ord('a') :
        rc.update_jog('jog_y_m')

    if key == ord('w') :
        rc.update_jog('jog_z_p')
        
    if key == ord('s') :
        rc.update_jog('jog_z_m')

    if key == ord('j') :
        rc.update_jog('jog_a_p')
    
    if key == ord('l') :
        rc.update_jog('jog_a_m')
    
    if key == ord('u') :
        rc.update_jog('jog_b_p')
    
    if key == ord('o') :
        rc.update_jog('jog_b_m')
    
    if key == ord('i') :
        rc.update_jog('jog_c_p')
    
    if key == ord('k') :
        rc.update_jog('jog_c_m')