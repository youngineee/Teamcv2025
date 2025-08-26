import cv2
import mediapipe as mp
import numpy as np
import time

# --- 설정값 ---
EAR_THRESHOLD = 0.2
BLINK_THRESHOLD = 20
BLINK_INTERVAL = 300
HORIZ_THRESHOLD = 20
VERT_THRESHOLD = 20
PARALLEL_THRESHOLD = 15

# --- 상태 초기화 ---
blink_count = 0
eye_closed = False
start_time = time.time()

# --- MediaPipe 초기화 ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

# --- 보조 함수 ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ang1 = np.arctan2(a[1]-b[1], a[0]-b[0])
    ang2 = np.arctan2(c[1]-b[1], c[0]-b[0])
    ang = np.abs(ang1 - ang2)
    ang = np.degrees(ang)
    return ang if ang <= 180 else 360 - ang

def angle_between(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    cos = np.clip(cos, -1, 1)
    return np.degrees(np.arccos(cos))

# --- 비디오 실행 ---
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results_pose = pose.process(image)
        results_face = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results_pose.pose_landmarks:
            lm = results_pose.pose_landmarks.landmark
            def get_xy(idx): return [lm[idx].x * w, lm[idx].y * h]

            left_shoulder = get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            right_shoulder = get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            nose = get_xy(mp_pose.PoseLandmark.NOSE.value)
            left_ear = get_xy(mp_pose.PoseLandmark.LEFT_EAR.value)
            right_ear = get_xy(mp_pose.PoseLandmark.RIGHT_EAR.value)
            left_hip = get_xy(mp_pose.PoseLandmark.LEFT_HIP.value)
            right_hip = get_xy(mp_pose.PoseLandmark.RIGHT_HIP.value)

            # 카메라 방향 판별
            view_mode = "unknown"
            dx_ear = abs(left_ear[0] - right_ear[0])
            if dx_ear > 100:
                view_mode = "front"
            else:
                view_mode = "side"

            cv2.putText(image, f"View: {view_mode}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if view_mode == "front":
                # 어깨 수평
                dy_sh = abs(left_shoulder[1] - right_shoulder[1])
                text = "Shoulders Level" if dy_sh < HORIZ_THRESHOLD else "Shoulders Tilted"
                cv2.putText(image, text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if dy_sh < HORIZ_THRESHOLD else (0,0,255), 2)

                # 코 중앙
                center_x = (left_shoulder[0] + right_shoulder[0]) / 2
                dx_nc = abs(nose[0] - center_x)
                text = "Neck Aligned" if dx_nc < VERT_THRESHOLD else "Neck Tilted"
                cv2.putText(image, text, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if dx_nc < VERT_THRESHOLD else (0,0,255), 2)

                # 귀 수평
                dy_ear = abs(left_ear[1] - right_ear[1])
                text = "Face Level" if dy_ear < HORIZ_THRESHOLD else "Face Tilted"
                cv2.putText(image, text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if dy_ear < HORIZ_THRESHOLD else (0,0,255), 2)

                # 눈 깜빡임
                if results_face.multi_face_landmarks:
                    flm = results_face.multi_face_landmarks[0].landmark
                    def fxy(i): return np.array([flm[i].x * w, flm[i].y * h])
                    p1,p2,p3,p5,p6,p4 = fxy(33), fxy(159), fxy(158), fxy(153), fxy(145), fxy(133)
                    ear_l = (np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / (2 * np.linalg.norm(p1-p4))
                    p1,p2,p3,p5,p6,p4 = fxy(263), fxy(386), fxy(385), fxy(380), fxy(374), fxy(362)
                    ear_r = (np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / (2 * np.linalg.norm(p1-p4))
                    ear = (ear_l + ear_r) / 2
                    if ear < EAR_THRESHOLD:
                        if not eye_closed: eye_closed = True
                    else:
                        if eye_closed:
                            blink_count += 1
                            eye_closed = False

                elapsed = time.time() - start_time
                if elapsed >= BLINK_INTERVAL and blink_count < BLINK_THRESHOLD:
                    blink_text = f"Blink:{blink_count} (<{BLINK_THRESHOLD})"
                    blink_color = (0, 0, 255)
                else:
                    blink_text = f"Blink:{blink_count}"
                    blink_color = (255, 255, 0)
                cv2.putText(image, blink_text, (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, blink_color, 2)

            elif view_mode == "side":
                shoulder = left_shoulder if left_shoulder[0] < right_shoulder[0] else right_shoulder
                hip = left_hip if left_hip[0] < right_hip[0] else right_hip
                dx_sp = abs(shoulder[0] - hip[0])
                text = "Spine Vertical" if dx_sp < VERT_THRESHOLD else "Spine Bent"
                cv2.putText(image, text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if dx_sp < VERT_THRESHOLD else (0,0,255), 2)
                # 초기 길이 측정 후 짧아질 시 "Spine Bent" 표시 추가
                
                v1 = np.array(hip) - np.array(shoulder)
                v2 = np.array(right_ear) - np.array(right_shoulder)
                tn_ang = angle_between(v1, v2)
                text = "Neck OK" if abs(tn_ang) < PARALLEL_THRESHOLD else "Turtle Neck"
                cv2.putText(image, text, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if abs(tn_ang) < PARALLEL_THRESHOLD else (0,0,255), 2)

            mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Posture Analysis", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
