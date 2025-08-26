import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# --- 설정값 ---
BASE_EAR_THRESHOLD = 0.20      # EAR 기본값(얼굴크기(IPD)로 0.15~0.28 사이로 자동 보정)
BLINK_THRESHOLD_PER_WIN = 20   # 깜빡임 윈도우 동안 필요한 횟수
BLINK_WINDOW_SEC = 300         # 깜빡임 집계 윈도우(초)

# 정면 수평 판단용 (IPD 비례 가변 임계값)
Y_THR_RATIO = 0.06             # 눈썹/눈/광대 y차이 임계 = IPD * 0.06 (대략)
X_THR_RATIO = 0.05             # 중앙 위/아래 x차이 임계 = IPD * 0.05
MIN_PX_THR  = 2.0              # 너무 작은 임계값 방지용 최소 px

EMA_ALPHA = 0.2                # 지표 지수평활 정도(0~1)

# --- 상태값 ---
blink_count = 0
eye_closed = False
win_start = time.time()
flip_view = False
ema_vals = {}

# --- MediaPipe 초기화 ---
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose  # (깜빡임/수평판단엔 불필요하지만, 원하시면 어깨 등도 표시 가능)

# --- 보조 함수 ---
def ema(key, value, alpha=EMA_ALPHA):
    if value is None:
        return None
    if key not in ema_vals:
        ema_vals[key] = value
    else:
        ema_vals[key] = alpha * value + (1 - alpha) * ema_vals[key]
    return ema_vals[key]

def safe_dist(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.linalg.norm(a - b))

def adaptive_thresh_by_ipd(ipd, ratio):
    if ipd is None or ipd <= 1:
        return max(MIN_PX_THR, 60.0 * ratio)  # IPD를 모르면 60px 가정
    return max(MIN_PX_THR, ipd * ratio)

def compute_ear(flm, w, h, left=True):
    def fxy(i): 
        return np.array([flm[i].x * w, flm[i].y * h], dtype=np.float32)
    if left:
        p1,p2,p3,p5,p6,p4 = fxy(33), fxy(159), fxy(158), fxy(153), fxy(145), fxy(133)
    else:
        p1,p2,p3,p5,p6,p4 = fxy(263), fxy(386), fxy(385), fxy(380), fxy(374), fxy(362)
    num = (np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5))
    den = (2.0 * np.linalg.norm(p1-p4))
    if den < 1e-6:
        return None
    ear = float(num / den)
    return float(np.clip(ear, 0.0, 1.0))

# --- 비디오 실행 ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

    fps_deque = deque(maxlen=20)
    last_time = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if flip_view:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        results_face = face_mesh.process(rgb)

        rgb.flags.writeable = True
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        ipd_px = None
        head_level = None
        detail_msgs = []

        if results_face.multi_face_landmarks:
            flm = results_face.multi_face_landmarks[0].landmark

            def p(idx):
                return np.array([flm[idx].x * w, flm[idx].y * h], dtype=np.float32)

            # --- 얼굴 크기: IPD(눈 사이 거리, 대략 바깥 눈꼬리 33-263) ---
            L_eye_outer = p(33)
            R_eye_outer = p(263)
            ipd_px = safe_dist(L_eye_outer, R_eye_outer)

            # --- 눈썹: 좌/우 대표점(105, 334) y차이 ---
            brow_L = p(105)
            brow_R = p(334)
            dy_brow = abs(brow_L[1] - brow_R[1])
            dy_brow_s = ema("dy_brow", dy_brow)
            y_thr = adaptive_thresh_by_ipd(ipd_px, Y_THR_RATIO)
            brow_ok = dy_brow_s is not None and dy_brow_s <= y_thr
            detail_msgs.append(f"Brow Δy={dy_brow_s:.1f}px thr~{y_thr:.1f} -> {'OK' if brow_ok else 'TILT'}")

            # --- 눈: 각 눈의 중앙(바깥/안쪽 눈꼬리 평균) y차이 ---
            L_eye_inner = p(133)
            R_eye_inner = p(362)
            L_eye_c = (L_eye_outer + L_eye_inner) / 2.0
            R_eye_c = (R_eye_outer + R_eye_inner) / 2.0
            dy_eye = abs(L_eye_c[1] - R_eye_c[1])
            dy_eye_s = ema("dy_eye", dy_eye)
            eye_ok = dy_eye_s is not None and dy_eye_s <= y_thr
            detail_msgs.append(f"Eyes Δy={dy_eye_s:.1f}px thr~{y_thr:.1f} -> {'OK' if eye_ok else 'TILT'}")

            # --- 광대(치크본): 외곽 윤곽의 좌/우 치크 근방(50, 280) y차이 ---
            cheek_L = p(50)
            cheek_R = p(280)
            dy_cheek = abs(cheek_L[1] - cheek_R[1])
            dy_cheek_s = ema("dy_cheek", dy_cheek)
            cheek_ok = dy_cheek_s is not None and dy_cheek_s <= y_thr
            detail_msgs.append(f"Cheek Δy={dy_cheek_s:.1f}px thr~{y_thr:.1f} -> {'OK' if cheek_ok else 'TILT'}")

            # --- 중앙 세로선: 이마(10) vs 턱(152) x좌표 차이 ---
            top_c = p(10)
            bot_c = p(152)
            dx_mid = abs(top_c[0] - bot_c[0])
            dx_mid_s = ema("dx_mid", dx_mid)
            x_thr = adaptive_thresh_by_ipd(ipd_px, X_THR_RATIO)
            mid_ok = dx_mid_s is not None and dx_mid_s <= x_thr
            detail_msgs.append(f"Midline Δx={dx_mid_s:.1f}px thr~{x_thr:.1f} -> {'OK' if mid_ok else 'TILT'}")

            # --- 최종 판단: 하나라도 OK면 수평 ---
            head_level = (brow_ok or eye_ok or cheek_ok or mid_ok)

            # --- EAR(깜빡임) 계산 ---
            ear_l = compute_ear(flm, w, h, left=True)
            ear_r = compute_ear(flm, w, h, left=False)
            ear = None
            if ear_l is not None and ear_r is not None:
                ear = (ear_l + ear_r) / 2.0
                ear = ema("ear", ear)

            ear_thr = BASE_EAR_THRESHOLD
            if ipd_px is not None and ipd_px > 1:
                ear_thr = np.clip(BASE_EAR_THRESHOLD * (ipd_px / 60.0), 0.15, 0.28)

            if ear is not None:
                if ear < ear_thr:
                    if not eye_closed:
                        eye_closed = True
                else:
                    if eye_closed:
                        blink_count += 1
                        eye_closed = False

        # --- 화면 표시 ---
        title = "Head Level" if head_level else "Head Tilted" if head_level is not None else "Face: N/A"
        color = (0, 200, 0) if head_level else ((0, 0, 255) if head_level is not None else (200, 200, 200))
        cv2.putText(image, title, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        y0 = 80
        for i, msg in enumerate(detail_msgs[:4]):
            cv2.putText(image, msg, (30, y0 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

        # 깜빡임 윈도우 표시/리셋
        now = time.time()
        elapsed = now - win_start
        if elapsed >= BLINK_WINDOW_SEC:
            pass_flag = (blink_count >= BLINK_THRESHOLD_PER_WIN)
            blink_text = f"Blink {blink_count} / {BLINK_THRESHOLD_PER_WIN} in {int(elapsed)}s"
            cv2.putText(image, blink_text, (30, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 200, 0) if pass_flag else (0, 0, 255), 2)
            blink_count = 0
            win_start = now
        else:
            remain = BLINK_WINDOW_SEC - int(elapsed)
            pass_flag = (blink_count >= BLINK_THRESHOLD_PER_WIN)
            blink_text = f"Blink {blink_count} / {BLINK_THRESHOLD_PER_WIN} (remain {remain}s)"
            cv2.putText(image, blink_text, (30, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 200, 0) if pass_flag else (255, 255, 0), 2)

        # FPS
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps_deque.append(1.0/dt)
            fps = np.mean(fps_deque)
            cv2.putText(image, f"FPS: {fps:.1f}", (w-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

        # 안내
        cv2.putText(image, "Keys: [r]=reset  [f]=flip  [q]=quit", (30, h-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

        cv2.imshow("Front Pose - Head Level (FaceMesh)", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            flip_view = not flip_view
        elif key == ord('r'):
            blink_count = 0
            win_start = time.time()
            ema_vals.clear()

cap.release()
cv2.destroyAllWindows()
