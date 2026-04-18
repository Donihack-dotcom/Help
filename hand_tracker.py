import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

# Модель баптаулары
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print("Қол трекері іске қосылды. Жабу үшін 'q' басыңыз.")

# Байланыс сызықтары
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # MediaPipe форматына өзгерту
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Қолды анықтау
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            # Нүктелерді алу
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            # Сызықтарды сызу
            for start, end in CONNECTIONS:
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

            # Нүктелерді сызу
            for i, (cx, cy) in enumerate(points):
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Сұқ саусақ ұшы (8) — үлкен нүкте
            cv2.circle(frame, points[8], 12, (0, 0, 255), -1)

        cv2.putText(frame, "Qol tabyldy!", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Qol joq...", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Qol Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
