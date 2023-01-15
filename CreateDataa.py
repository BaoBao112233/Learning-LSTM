import cv2
import mediapipe as mp
import pandas as pd

# Đọc ảnh từ Camera / Webcam
cap = cv2.VideoCapture(0)

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
posee = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# lưu trữ các thông số của khung xương
lm_list = []

def create_landmark_timestep(results):
    # results chứa các tọa độ các điểm trên khung xương
    print(results.pose_landmarks.landmark)
    return 1 

def draw_landmark_on_image(mpDraw, results, frame):
    pass

while True:
    ret, frame = cap.read()
    if ret:
        # Nhận diện các poses
        frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # Frame được đọc từ cv2 là BGR
        results = posee.process(frameRGB)
        if results.pose_landmarks:
            # ghi nhận thông số của khung xương
            lm = create_landmark_timestep(results)
            lm_list.append(lm)
            # vẽ khung xương lên ảnh được đọc    
            # frame = draw_landmark_on_image(mpDraw, results, frame)


        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'): # Set up exit button
            break

cap.release()
cv2.destroyAllWindows()
