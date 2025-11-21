import cv2
import dlib

# Load detector
detector = dlib.get_frontal_face_detector()

# Start webcam
cap = cv2.VideoCapture(0)    # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector(rgb, 1)

    # Draw rectangles
    for d in faces:
        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show output
    cv2.imshow("Live Face Detection (HOG + dlib)", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
