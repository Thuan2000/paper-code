import cv2
from rabbitmq import RabbitMQ
from cv_utils import encode_image

rb = RabbitMQ()
cap = cv2.VideoCapture(0)
frame_counter = 0
while (True):
    frame_counter += 1
    print(frame_counter)
    _, frame = cap.read()
    bin_image = encode_image(frame)
    rb.send('uob-live', bin_image)
