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

label = "Lac_Dau"
num_of_frame = 60 #600
count = 0

def create_landmark_timestep(results):
    # results chứa các tọa độ các điểm trên khung xương
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    # Vẽ các đường nối giữa các điểm để tạo ra khung xương 
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    
    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = frame.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx,cy), 5, (0, 0, 255), cv2.FILLED)
    return frame

while len(lm_list) <= num_of_frame:
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
            frame = draw_landmark_on_image(mpDraw, results, frame)


        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'): # Set up exit button
            break

# Viết là file csv
df =  pd.DataFrame(lm_list)
df.to_csv(label + ".txt")

cap.release()
cv2.destroyAllWindows()
