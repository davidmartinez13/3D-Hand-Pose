from opcua import ua
from opcua import Client
import cv2
import numpy as np
import time
class OPCUA_Client:
    def __init__(self):
        print('[INFO]:Client object started.')

    def connect_OPCUA_server(self):
        """
        Connects OPC UA Client to Server on PLC.

        """
        password = ""
        self.client = Client("opc.tcp://user:"+str(password)+"@00.00.00.000:0000/")
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

    def update_jog(self, jog_dict):
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

        self.Jog_X_M.set_value(ua.DataValue(jog_dict['jog_x_m']))
        self.Jog_X_P.set_value(ua.DataValue(jog_dict['jog_x_p']))
        self.Jog_Y_M.set_value(ua.DataValue(jog_dict['jog_y_m']))
        self.Jog_Y_P.set_value(ua.DataValue(jog_dict['jog_y_p']))
        self.Jog_Z_M.set_value(ua.DataValue(jog_dict['jog_z_m']))
        self.Jog_Z_P.set_value(ua.DataValue(jog_dict['jog_z_p']))

        self.Jog_A_M.set_value(ua.DataValue(jog_dict['jog_a_m']))
        self.Jog_A_P.set_value(ua.DataValue(jog_dict['jog_a_p']))
        self.Jog_B_M.set_value(ua.DataValue(jog_dict['jog_b_m']))
        self.Jog_B_P.set_value(ua.DataValue(jog_dict['jog_b_p']))
        self.Jog_C_M.set_value(ua.DataValue(jog_dict['jog_c_m']))
        self.Jog_C_P.set_value(ua.DataValue(jog_dict['jog_c_p']))
        print(jog_dict)
        time.sleep(0.3)
def set_jog_keys(jog_dict,jog_key):
    for key in jog_dict:
        if key is jog_key:
            jog_dict[key]= not jog_dict[key]
        else:
            jog_dict[key] = False

cl = OPCUA_Client()

cl.connect_OPCUA_server()
gripper = False
jog_x_p = False
jog_x_m = False
jog_y_p = False
jog_y_m = False
jog_z_p = False
jog_z_m = False

jog_a_p = False
jog_a_m = False
jog_b_p = False
jog_b_m = False
jog_c_p = False
jog_c_m = False

jog_dict = {
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
while True:
    # jog_x_p = False
    cl.get_nodes()
    x_pos, y_pos, z_pos, a_pos, b_pos, c_pos, status_pos, turn_pos = cl.get_actual_pos()

    screen = np.zeros((960,1280))
    

    cv2.putText(screen,'x:'+ str(x_pos),(60,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(screen,'y:'+ str(y_pos),(60,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(screen,'z:'+ str(z_pos),(60,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(screen,'a:'+ str(a_pos),(60,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(screen,'b:'+ str(b_pos),(60,110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(screen,'c:'+ str(c_pos),(60,130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(screen,'Status:'+ str(status_pos),(60,150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(screen,'Turn:'+ str(turn_pos),(60,170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    # print(jog_a_p, jog_a_m, jog_b_p, jog_b_m, jog_c_p, jog_c_m)    
    cv2.imshow("Frame", screen)
    key = cv2.waitKey(1)
    if key == 27:
        cl.client.disconnect()
        break
    if key == ord('g'):
        gripper = not gripper
        cl.Gripper_State.set_value(ua.DataValue(gripper))
        time.sleep(0.4)

    if key == ord('e') :
        set_jog_keys(jog_dict, 'jog_x_p')
        cl.update_jog(jog_dict)
        
    if key == ord('q') :
        set_jog_keys(jog_dict, 'jog_x_m')
        cl.update_jog(jog_dict)
        
    if key == ord('d') :
        set_jog_keys(jog_dict, 'jog_y_p')
        cl.update_jog(jog_dict)

    if key == ord('a') :
        set_jog_keys(jog_dict, 'jog_y_m')
        cl.update_jog(jog_dict)

    if key == ord('w') :
        set_jog_keys(jog_dict, 'jog_z_p')
        cl.update_jog(jog_dict)
        
    if key == ord('s') :
        set_jog_keys(jog_dict, 'jog_z_m')
        cl.update_jog(jog_dict)

    if key == ord('j') :
        set_jog_keys(jog_dict, 'jog_a_p')
        cl.update_jog(jog_dict)
    
    if key == ord('l') :
        set_jog_keys(jog_dict, 'jog_a_m')
        cl.update_jog(jog_dict)
    
    if key == ord('u') :
        set_jog_keys(jog_dict, 'jog_b_p')
        cl.update_jog(jog_dict)
    
    if key == ord('o') :
        set_jog_keys(jog_dict, 'jog_b_m')
        cl.update_jog(jog_dict)
    
    if key == ord('i') :
        set_jog_keys(jog_dict, 'jog_c_p')
        cl.update_jog(jog_dict)
    
    if key == ord('k') :
        set_jog_keys(jog_dict, 'jog_c_m')
        cl.update_jog(jog_dict)