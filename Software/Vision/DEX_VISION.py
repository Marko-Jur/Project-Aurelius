import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import math 

def vector_angle (vector_a, vector_b) :
     dot_product = 0;
     for i in range(len(vector_a)):
          dot_product += vector_a[i]*vector_b[i]
     cos_t = dot_product/(np.linalg.norm(vector_a)*np.linalg.norm(vector_b))
     return math.acos(cos_t)*(180/math.pi)

if __name__ == "__main__":

     mp_draw = mp.solutions.drawing_utils
     mp_holistic = mp.solutions.holistic

     right_hand_coordinates = np.zeros((21, 3), dtype = float)
     left_hand_coordinates = np.zeros((21, 3), dtype = float)

     cam = cv2.VideoCapture(0)
     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
          while cam.isOpened():
               success, image = cam.read()
               image_y, image_x, _ = image.shape
               scale_z = 250
               #image_y = 1
               #image_x = 1
               #scale_z = 1
               
               image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

               holistic_data = holistic.process(image)

               image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    
               mp_draw.draw_landmarks(image, holistic_data.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                                      mp_draw.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
               
               mp_draw.draw_landmarks(image, holistic_data.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                                      mp_draw.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))

               '''
               mp_draw.draw_landmarks(image, holistic_data.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_draw.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=4),
                                      mp_draw.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))

               mp_draw.draw_landmarks(image, holistic_data.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                      mp_draw.DrawingSpec(color=(255,100,100), thickness=2, circle_radius=4),
                                      mp_draw.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))

               
               if holistic_data.left_hand_landmarks:
                    for num, left_hand in enumerate(holistic_data.left_hand_landmarks):
                         mp_draw.draw_landmarks(image, left_hand, mp_holistic.HAND_CONNECTIONS,
                                      mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                                      mp_draw.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
               if holistic_data.right_hand_landmarks:     
                    for num, right_hand in enumerate(holistic_data.right_hand_landmarks):
                         mp_draw.draw_landmarks(image, right_hand, mp_holistic.HAND_CONNECTIONS,
                                      mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                                      mp_draw.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
               '''


               mp_draw.draw_landmarks(image, holistic_data.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_draw.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=4),
                                      mp_draw.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
               mp_draw.draw_landmarks(image, holistic_data.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                      mp_draw.DrawingSpec(color=(255,100,100), thickness=2, circle_radius=4),
                                      mp_draw.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
               
               if holistic_data.right_hand_landmarks:
                    right_hand_coordinates[0] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].z)*scale_z]
                    right_hand_coordinates[1] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].z)*scale_z]
                    right_hand_coordinates[2] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].z)*scale_z]
                    right_hand_coordinates[3] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].z)*scale_z]
                    right_hand_coordinates[4] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].z)*scale_z]
                    right_hand_coordinates[5] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].z)*scale_z]
                    right_hand_coordinates[6] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].z)*scale_z]
                    right_hand_coordinates[7] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].z)*scale_z]
                    right_hand_coordinates[8] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].z)*scale_z]
                    right_hand_coordinates[9] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].z)]
                    right_hand_coordinates[10] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].z)]
                    right_hand_coordinates[11] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].z)]
                    right_hand_coordinates[12] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].z)]
                    right_hand_coordinates[13] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].z)]
                    right_hand_coordinates[14] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].z)]
                    right_hand_coordinates[15] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].z)]
                    right_hand_coordinates[16] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].z)]
                    right_hand_coordinates[17] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].z)]
                    right_hand_coordinates[18] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].z)]
                    right_hand_coordinates[19] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].z)]
                    right_hand_coordinates[20] = [(holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].x)*image_x,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].y)*image_y,
                                                 (holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].z)]
                    cv2.putText(image, 'Right Hand', tuple([int(right_hand_coordinates[0][0]), int(right_hand_coordinates[0][1])]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    index_finger_angle = vector_angle(np.subtract(right_hand_coordinates[5], right_hand_coordinates[0]), np.subtract(right_hand_coordinates[6], right_hand_coordinates[5]))
                    #index_finger_angle = vector_angle(np.subtract([50, 50, 0], [0, 0, 0]), np.subtract([120, 70, 0], [50, 50, 0]))
                    cv2.putText(image, str(index_finger_angle), tuple([50, 50]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    
                    
               
               

               if holistic_data.left_hand_landmarks:
                    left_hand_coordinates[0] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].z)]
                    left_hand_coordinates[1] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].z)]
                    left_hand_coordinates[2] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].z)]
                    left_hand_coordinates[3] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].z)]
                    left_hand_coordinates[4] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].z)]
                    left_hand_coordinates[5] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].z)]
                    left_hand_coordinates[6] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].z)]
                    left_hand_coordinates[7] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].z)]
                    left_hand_coordinates[8] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].z)]
                    left_hand_coordinates[9] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].z)]
                    left_hand_coordinates[10] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].z)]
                    left_hand_coordinates[11] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].z)]
                    left_hand_coordinates[12] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].z)]
                    left_hand_coordinates[13] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].z)]
                    left_hand_coordinates[14] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].z)]
                    left_hand_coordinates[15] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].z)]
                    left_hand_coordinates[16] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].z)]
                    left_hand_coordinates[17] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].z)]
                    left_hand_coordinates[18] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].z)]
                    left_hand_coordinates[19] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].z)]
                    left_hand_coordinates[20] = [(holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].x)*image_x,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].y)*image_y,
                                                 (holistic_data.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].z)]
                    cv2.putText(image, 'Left Hand', tuple([int(left_hand_coordinates[0][0]), int(left_hand_coordinates[0][1])]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                                             
               cv2.imshow('Holistic Feed', image)
               '''
               if holistic_data.right_hand_landmarks:
                    
                    print('Right hand wrist: ',
                         'x: ', {holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x*image_x},
                         ', y: ', {holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y*image_y},
                         ', z: ', {holistic_data.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].z})

               '''
               print(right_hand_coordinates[0])

               if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

     cap.release()
     cv2.destroyAllWindows()


