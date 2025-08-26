import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# --- 설정값 ---
BASE_EAR_THRESHOLD = 0.20
BLINK_THRESHOLD_PER_WIN = 20
BLINK_WINDOW_SEC = 300

# 정면 수평 판단용 (IPD 비례 가변 임계값)
Y_THR_RATIO_BROW  = 0.06   # 눈썹 Δy
Y_THR_RATIO_EYE   = 0.05   # 눈 Δy
Y_THR_RATIO_CHEEK = 0.06   # 광대 Δy
X_THR_RATIO_MID   = 0.05   # 중앙선 Δx
MIN_PX_THR  = 2.0

EMA_ALPHA = 0.2

# --- 상태값 ---
blink_count = 0
eye_closed = False
win_start = time.time()
flip_view = False
ema_vals = {}

# --- MediaPipe ---
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

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

def adaptive_thresh(ipd, ratio):
    if ipd is None or ipd <= 1:
        return max(MIN_PX_THR, 60.0 * ratio)
    return max(MIN_PX_THR, ipd * ratio)

def compute_ear(flm, w, h, left=True):
    def fxy(i):
        return np.array([flm[i].x * w, flm[i].y * h], dtype=np.float32)
    if left:
        p1,p2,p3,p5,p6,p4 = fxy(33), fxy(159), fxy(158), fxy(153), fxy(145), fxy(133)
    else:
        p1,p2,p3,p5,p6,p4 = fxy(263), fxy(386), fxy(385), fxy(380), fxy(374), fxy(362)
    den = 2.0 * np.linalg.norm(p1-p4)
    if den < 1e-6: return None
    ear = (np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / den
    return float(np.clip(ear, 0.0, 1.0))

def draw_marker(img, pt, color, r=4, thick=2):
    if pt is None: return
    cv2.circle(img, (int(pt[0]), int(pt[1])), r, color, thick, lineType=cv2.LINE_AA)

def draw_line(img, p1, p2, color, thick=2):
    if p1 is None or p2 is None: return
    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thick, lineType=cv2.LINE_AA)

def put_text(img, text, org, color, scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_panel(img, x, y, w, h, alpha=0.35):
    overlay = img.copy()
    cv2.rectangle(overlay, (x,y), (x+w, y+h), (20,20,20), -1)
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

# --- 메인 ---
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

        # 기본 색 정의
        GREEN = (0, 210, 0)
        RED   = (0, 0, 230)
        YEL   = (0, 220, 220)
        WHITE = (230, 230, 230)
        GRAY  = (160, 160, 160)
        CYAN  = (200, 255, 255)
        BLUE  = (190, 160, 0)

        ipd_px = None
        head_level = None
        detail = {}

        if results_face.multi_face_landmarks:
            flm = results_face.multi_face_landmarks[0].landmark
            def p(i): return np.array([flm[i].x * w, flm[i].y * h], dtype=np.float32)

            # 포인트 추출
            L_eye_outer, R_eye_outer = p(33), p(263)
            L_eye_inner, R_eye_inner = p(133), p(362)
            brow_L, brow_R = p(105), p(334)
            cheek_L, cheek_R = p(50), p(280)
            top_c, bot_c = p(10), p(152)

            # IPD
            ipd_px = safe_dist(L_eye_outer, R_eye_outer)

            # 임계값
            thr_brow  = adaptive_thresh(ipd_px, Y_THR_RATIO_BROW)
            thr_eye   = adaptive_thresh(ipd_px, Y_THR_RATIO_EYE)
            thr_cheek = adaptive_thresh(ipd_px, Y_THR_RATIO_CHEEK)
            thr_mid   = adaptive_thresh(ipd_px, X_THR_RATIO_MID)

            # 지표 계산 + EMA
            dy_brow  = abs(brow_L[1] - brow_R[1]);   dy_brow_s  = ema("dy_brow", dy_brow)
            dy_eye   = abs(((L_eye_outer+L_eye_inner)/2)[1] - ((R_eye_outer+R_eye_inner)/2)[1]); dy_eye_s = ema("dy_eye", dy_eye)
            dy_cheek = abs(cheek_L[1] - cheek_R[1]); dy_cheek_s = ema("dy_cheek", dy_cheek)
            dx_mid   = abs(top_c[0] - bot_c[0]);     dx_mid_s   = ema("dx_mid", dx_mid)

            brow_ok  = dy_brow_s  is not None and dy_brow_s  <= thr_brow
            eye_ok   = dy_eye_s   is not None and dy_eye_s   <= thr_eye
            cheek_ok = dy_cheek_s is not None and dy_cheek_s <= thr_cheek
            mid_ok   = dx_mid_s   is not None and dx_mid_s   <= thr_mid
            head_level = (brow_ok or eye_ok or cheek_ok or mid_ok)

            # -------- 보조선/마커 시각화 --------
            # 눈썹 라인
            draw_line(image, brow_L, brow_R, GREEN if brow_ok else RED, 3)
            draw_marker(image, brow_L, BLUE, r=4); draw_marker(image, brow_R, BLUE, r=4)
            put_text(image, f"Brows dy={dy_brow_s:.1f}/{thr_brow:.1f}px", (30, 80),
                     GREEN if brow_ok else RED, 0.65, 2)

            # 눈 중앙 라인
            L_eye_c = (L_eye_outer + L_eye_inner)/2.0
            R_eye_c = (R_eye_outer + R_eye_inner)/2.0
            draw_line(image, L_eye_c, R_eye_c, GREEN if eye_ok else RED, 3)
            draw_marker(image, L_eye_c, CYAN, r=4); draw_marker(image, R_eye_c, CYAN, r=4)
            put_text(image, f"Eyes  dy={dy_eye_s:.1f}/{thr_eye:.1f}px", (30, 110),
                     GREEN if eye_ok else RED, 0.65, 2)

            # 광대 라인
            draw_line(image, cheek_L, cheek_R, GREEN if cheek_ok else RED, 3)
            draw_marker(image, cheek_L, (0,180,255), r=4); draw_marker(image, cheek_R, (0,180,255), r=4)
            put_text(image, f"Cheek dy={dy_cheek_s:.1f}/{thr_cheek:.1f}px", (30, 140),
                     GREEN if cheek_ok else RED, 0.65, 2)

            # 중앙 세로선
            draw_line(image, top_c, bot_c, GREEN if mid_ok else RED, 3)
            draw_marker(image, top_c, YEL, r=5); draw_marker(image, bot_c, YEL, r=5)
            put_text(image, f"Mid   dx={dx_mid_s:.1f}/{thr_mid:.1f}px", (30, 170),
                     GREEN if mid_ok else RED, 0.65, 2)

            # EAR / 깜빡임
            ear_l = compute_ear(flm, w, h, left=True)
            ear_r = compute_ear(flm, w, h, left=False)
            ear = None
            if ear_l is not None and ear_r is not None:
                ear = ema("ear", (ear_l + ear_r)/2.0)

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

        # -------- 상단 상태 배지 --------
        title = "Head Level" if head_level else "Head Tilted" if head_level is not None else "Face: N/A"
        title_color = (0, 200, 0) if head_level else ((0, 0, 230) if head_level is not None else (200,200,200))
        put_text(image, title, (30, 40), title_color, 1.0, 2)

        # -------- 우측 패널: 요약/가이드 --------
        panel_w = 300
        image = draw_panel(image, w - panel_w - 20, 20, panel_w, 170, 0.35)
        px, py = w - panel_w - 10, 40
        put_text(image, "Summary", (px, py), WHITE, 0.8, 2); py += 28
        put_text(image, "Green = within threshold", (px, py), (180,255,180), 0.6, 2); py += 22
        put_text(image, "Red   = tilted", (px, py), (150,150,255), 0.6, 2); py += 22
        put_text(image, "Keys: [f] flip  [r] reset  [q] quit", (px, py), GRAY, 0.55, 2); py += 22

        # 깜빡임 표시(좌측 하단)
        now = time.time()
        elapsed = now - win_start
        if elapsed >= BLINK_WINDOW_SEC:
            pass_flag = (blink_count >= BLINK_THRESHOLD_PER_WIN)
            msg = f"Blink {blink_count}/{BLINK_THRESHOLD_PER_WIN} in {int(elapsed)}s"
            put_text(image, msg, (30, h - 60), (0, 220, 0) if pass_flag else RED, 0.75, 2)
            blink_count = 0
            win_start = now
        else:
            remain = BLINK_WINDOW_SEC - int(elapsed)
            pass_flag = (blink_count >= BLINK_THRESHOLD_PER_WIN)
            msg = f"Blink {blink_count}/{BLINK_THRESHOLD_PER_WIN} (remain {remain}s)"
            put_text(image, msg, (30, h - 60), (0, 220, 0) if pass_flag else (255, 255, 0), 0.75, 2)

        # FPS
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps_deque.append(1.0/dt)
            fps = np.mean(fps_deque)
            put_text(image, f"FPS: {fps:.1f}", (w - 130, h - 20), (200,200,200), 0.8, 2)

        cv2.imshow("Front Pose - Head Level (with Guides)", image)
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
