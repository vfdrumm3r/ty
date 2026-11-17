
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    hands.close()
    exit()

print("Hand identification test started.")
print("Show your LEFT hand, then RIGHT hand.")
print("Press Q to quit.")

try:
    while True:

        success, frame = camera.read()

        if not success:
            break

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:

            for index, hand_landmarks in enumerate(
                results.multi_hand_landmarks
            ):

                label = (
                    results.multi_handedness[index]
                    .classification[0].label
                )

                confidence = (
                    results.multi_handedness[index]
                    .classification[0].score
                )

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                wrist = hand_landmarks.landmark[0]

                x = int(wrist.x * frame.shape[1])
                y = int(wrist.y * frame.shape[0])

                cv2.putText(
                    frame,
                    f"{label} ({confidence:.2f})",
                    (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("ISL Hand Identification Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    hands.close()

    print("Test finished.")
