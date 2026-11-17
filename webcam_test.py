
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import re

from collections import deque
from datetime import datetime


# ==========================================
# CONFIGURATION
# ==========================================

CAMERA_INDEX = 0

SAMPLE_FPS = 10
SAMPLE_INTERVAL = 1.0 / SAMPLE_FPS

SEQUENCE_DURATION = 2
SEQUENCE_LENGTH = SAMPLE_FPS * SEQUENCE_DURATION

FEATURES_PER_HAND = 21 * 3
FEATURES_PER_FRAME = FEATURES_PER_HAND * 2

DATASET_FOLDER = "dataset"

# Change these for each sign
CURRENT_LABEL = "HELLO"

# 1 = One-handed sign
# 2 = Two-handed sign
EXPECTED_HANDS = 1

SAVE_COOLDOWN = 2.0


# ==========================================
# INITIALIZE BUFFERS
# ==========================================

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
hand_count_buffer = deque(maxlen=SEQUENCE_LENGTH)

last_sample_time = time.perf_counter()
last_save_time = 0

save_count = 0

save_message = ""
save_message_until = 0


# ==========================================
# MEDIAPIPE INITIALIZATION
# ==========================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# LANDMARK EXTRACTION
# ==========================================

def extract_landmarks(results):

    left_hand = [0.0] * FEATURES_PER_HAND
    right_hand = [0.0] * FEATURES_PER_HAND

    if results.multi_hand_landmarks:

        for index, hand_landmarks in enumerate(
            results.multi_hand_landmarks
        ):

            hand_label = (
                results.multi_handedness[index]
                .classification[0].label
            )

            landmarks = []

            for landmark in hand_landmarks.landmark:
                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            if hand_label == "Left":
                left_hand = landmarks

            elif hand_label == "Right":
                right_hand = landmarks

    return left_hand + right_hand


# ==========================================
# SAVE SEQUENCE
# ==========================================

def save_sequence():

    global save_count
    global last_save_time
    global save_message
    global save_message_until

    current_time = time.perf_counter()

    if current_time - last_save_time < SAVE_COOLDOWN:
        return

    if len(sequence_buffer) < SEQUENCE_LENGTH:
        save_message = "Collecting frames... wait"
        save_message_until = current_time + 2
        return

    if len(hand_count_buffer) < SEQUENCE_LENGTH:
        return

    # Every frame must have the expected number of hands
    if any(count != EXPECTED_HANDS for count in hand_count_buffer):

        save_message = "Incorrect hand count in sequence!"
        save_message_until = current_time + 2

        print("Sequence rejected: incorrect hand count.")
        return

    label = re.sub(
        r"[^A-Za-z0-9_-]",
        "",
        CURRENT_LABEL.strip().upper()
    )

    if not label:
        save_message = "Invalid label!"
        save_message_until = current_time + 2
        return

    label_folder = os.path.join(DATASET_FOLDER, label)
    os.makedirs(label_folder, exist_ok=True)

    sequence = np.array(
        sequence_buffer,
        dtype=np.float32
    )

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURES_PER_FRAME
    ):
        save_message = "Invalid sequence shape!"
        save_message_until = current_time + 2
        return

    if not np.isfinite(sequence).all():
        save_message = "Invalid landmark values!"
        save_message_until = current_time + 2
        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = f"{label}_{timestamp}.npz"

    filepath = os.path.join(
        label_folder,
        filename
    )

    np.savez_compressed(
        filepath,
        sequence=sequence,
        label=label,
        expected_hands=EXPECTED_HANDS,
        sample_fps=SAMPLE_FPS,
        sequence_duration=SEQUENCE_DURATION
    )

    save_count += 1
    last_save_time = current_time

    # Clear buffers after successful saving
    sequence_buffer.clear()
    hand_count_buffer.clear()

    save_message = f"Saved {label} successfully!"
    save_message_until = current_time + 2

    print("\nSequence saved successfully!")
    print("Label:", label)
    print("Expected hands:", EXPECTED_HANDS)
    print("Shape:", sequence.shape)
    print("Location:", filepath)
    print("Total saved:", save_count)


# ==========================================
# CAMERA INITIALIZATION
# ==========================================

camera = cv2.VideoCapture(CAMERA_INDEX)

camera.set(cv2.CAP_PROP_FPS, 30)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    hands.close()
    exit()


print("\n================================")
print("ISL DATA COLLECTION STARTED")
print("================================")
print("Current label:", CURRENT_LABEL)
print("Expected hands:", EXPECTED_HANDS)
print("Press S to save a sequence.")
print("Press Q to quit.")
print("================================\n")


# ==========================================
# MAIN LOOP
# ==========================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Error reading camera.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        # Draw detected hand landmarks
        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        current_time = time.perf_counter()

        # Sample approximately 10 times per second
        if current_time - last_sample_time >= SAMPLE_INTERVAL:

            landmarks = extract_landmarks(results)

            detected_hands = (
                len(results.multi_hand_landmarks)
                if results.multi_hand_landmarks
                else 0
            )

            sequence_buffer.append(landmarks)
            hand_count_buffer.append(detected_hands)

            last_sample_time = current_time

        # ==================================
        # DISPLAY INFORMATION
        # ==================================

        detected_count = (
            len(results.multi_hand_landmarks)
            if results.multi_hand_landmarks
            else 0
        )

        cv2.putText(
            frame,
            f"Hands detected: {detected_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Expected hands: {EXPECTED_HANDS}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Label: {CURRENT_LABEL}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Saved sequences: {save_count}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Display save message
        if current_time < save_message_until:

            cv2.putText(
                frame,
                save_message,
                (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "ISL Data Collection",
            frame
        )

        # ==================================
        # KEYBOARD CONTROLS
        # ==================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") or key == ord("S"):
            save_sequence()

        elif key == ord("q") or key == ord("Q"):
            break


finally:

    camera.release()
    cv2.destroyAllWindows()
    hands.close()

    print("\nCamera closed.")
    print("Data collection stopped.")
