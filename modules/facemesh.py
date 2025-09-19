 
import cv2
#import mediapipe as mp
import math

 

def detect_head_movement(top, bottom):

    radians = math.atan2(bottom[1] - top[1], bottom[0] - top[0])
    degrees = math.degrees(radians)

    #Angulo de deteccion de 70 a 110 (-1 a 1)
    min_degrees = 70
    max_degrees = 110
    degree_range = max_degrees - min_degrees
    
    if degrees < min_degrees: degrees = min_degrees
    if degrees > max_degrees: degrees = max_degrees

    movement = ( ((degrees-min_degrees) / degree_range) * 2) - 1
    
    return degrees, movement

 
