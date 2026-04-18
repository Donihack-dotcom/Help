import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import os
import webbrowser

MODEL_PATH = "hand_landmarker.task"
TRIGGER_FRAMES = 15

# Қимыл → Видео
GESTURES = {
    "thumbs_up": "videos/uagyz.mp4",
    "fist":      "videos/hand.mp4",
}

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]


def is_thumbs_up(hand):
    # Бас бармақ жоғары, қалған 4 саусақ жиылған
    thumb_up = (hand[4].y < hand[5].y and
                hand[4].y < hand[9].y and
                hand[4].y < hand[13].y and
                hand[4].y < hand[17].y)
    index_curled  = hand[8].y  > hand[5].y
    middle_curled = hand[12].y > hand[9].y
    ring_curled   = hand[16].y > hand[13].y
    pinky_curled  = hand[20].y > hand[17].y
    return thumb_up and index_curled and middle_curled and ring_curled and pinky_curled


def is_fist(hand):
    # Барлық 5 саусақ жиылған — жұдырық ✊
    thumb_curled  = hand[4].y  > hand[3].y
    index_curled  = hand[8].y  > hand[5].y
    middle_curled = hand[12].y > hand[9].y
    ring_curled   = hand[16].y > hand[13].y
    pinky_curled  = hand[20].y > hand[17].y
    return thumb_curled and index_curled and middle_curled and ring_curled and pinky_curled


def is_peace(hand):
    # ✌️ Сұқ + ортаңғы саусақ ашық, қалғаны жабық
    index_open    = hand[8].y  < hand[5].y
    middle_open   = hand[12].y < hand[9].y
    ring_curled   = hand[16].y > hand[13].y
    pinky_curled  = hand[20].y > hand[17].y
    thumb_curled  = hand[4].y  > hand[3].y
    return index_open and middle_open and ring_curled and pinky_curled and thumb_curled


def detect_gesture(hand):
    if is_thumbs_up(hand):
        return "thumbs_up"
    if is_fist(hand):
        return "fist"
    if is_peace(hand):
        return "peace"
    return None


def play_video(path):
    abs_path = os.path.abspath(path)
    print(f"Видео ойнатылуда: {abs_path}")
    os.startfile(abs_path)


def open_youtube():
    print("YouTube ашылуда...")
    webbrowser.open("https://www.youtube.com")


base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
counter = 0
current_gesture = None

print("Іске қосылды!")
print("👍 Бас бармақ жоғары  → uagyz.mp4")
print("✊ Жұдырық            → hand.mp4")
print("✌️  Екі саусақ         → YouTube")
print("Жабу үшін 'q' басыңыз")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    gesture = None

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            for s, e in CONNECTIONS:
                cv2.line(frame, pts[s], pts[e], (0, 255, 0), 2)
            for cx, cy in pts:
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            gesture = detect_gesture(hand)

        # Қимыл анықталса
        if gesture and gesture == current_gesture:
            counter += 1
        elif gesture:
            current_gesture = gesture
            counter = 1
        else:
            counter = 0
            current_gesture = None

        # Экранда көрсету
        if gesture == "thumbs_up":
            label = f"THUMBS UP {counter}/{TRIGGER_FRAMES}"
            color = (0, 255, 0)
        elif gesture == "fist":
            label = f"FIST {counter}/{TRIGGER_FRAMES}"
            color = (0, 165, 255)
        elif gesture == "peace":
            label = f"PEACE - YouTube {counter}/{TRIGGER_FRAMES}"
            color = (255, 0, 255)
        else:
            label = "Qimyl joq"
            color = (0, 0, 255)

        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Прогресс жолағы
        if counter > 0:
            bar_color = (0, 255, 255) if gesture == "thumbs_up" else (0, 165, 255)
            bar_w = int((counter / TRIGGER_FRAMES) * w)
            cv2.rectangle(frame, (0, h - 30), (bar_w, h), bar_color, -1)

    else:
        counter = 0
        current_gesture = None
        cv2.putText(frame, "Qol joq...", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

    cv2.imshow("Gesture Player", frame)

    if counter >= TRIGGER_FRAMES:
        counter = 0
        if current_gesture == "peace":
            open_youtube()
        elif current_gesture in GESTURES:
            play_video(GESTURES[current_gesture])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
