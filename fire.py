import cv2
import math
import pygame
from ultralytics import YOLO

pygame.mixer.init()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


model = YOLO('best.pt')


classnames = ['fire', 'smoke']


def play_alarm_sound():
    try:
        pygame.mixer.music.load('emergency-alarm-69780.mp3')
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing sound: {e}")


alarm_triggered = False


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break


    frame = cv2.resize(frame, (1280, 640))
    result = model(frame, stream=True)


    fire_detected = False

    for info in result:
        boxes = info.boxes
        for box in boxes:

            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            class_id = int(box.cls[0].item())


            if confidence > 68:
                if classnames[class_id] == 'fire' or classnames[class_id] == 'smoke':
                    fire_detected = True
                    print(f"{classnames[class_id].capitalize()} detected with {confidence}% confidence.")


                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{classnames[class_id]} {confidence}%',
                                (int(x1), int(y1) - 10),
                                cv2.QT_FONT_BLACK,
                                0.9, (0, 255, 0), 2)

    # Play alarm sound if fire or smoke is detected
    if fire_detected and not alarm_triggered:
        play_alarm_sound()
        alarm_triggered = True


    if not fire_detected:
        alarm_triggered = False


    cv2.imshow('Fire and Smoke Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resourcesqqqqq
cap.release()
cv2.destroyAllWindows()
