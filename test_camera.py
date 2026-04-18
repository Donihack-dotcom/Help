import cv2

cap = cv2.VideoCapture(0)

print("Камера іске қосылды. Жабу үшін 'q' басыңыз.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Камера табылмады!")
        break

    cv2.imshow("Камера тесті", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Камера жабылды.")
