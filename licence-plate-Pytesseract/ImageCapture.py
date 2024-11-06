import cv2
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
buzz_pin = 18
barrier = 17

#GPIO.setup(led_pin, GPIO.OUT)


from PlateExtraction import extraction
from OpticalCharacterRecognition import ocr
from OpticalCharacterRecognition import check_if_string_in_file

image = cv2.imread('./CarPictures/004.jpg')
#live feed
#image = cv2.imread('source = 0')
cv2.imshow("loadimage", image)
plate = extraction(image)
cv2.imshow('frame',plate)

text = ocr(plate)
text = ''.join(e for e in text if e.isalnum())
print(text, end=" ")
if check_if_string_in_file('./Database/Database.txt', text) and text != "":
    print('The Car is Registered: ')
    print("Opening the Barrier:")
    # control your barrier 
else:
    print("Not Registered")