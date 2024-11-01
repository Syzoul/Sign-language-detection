import cv2
import mediapipe as mp
import numpy as np
import pickle
import os


# Hàm để lấy tọa độ các điểm landmarks của bàn tay phải
def get_hand_landmarks(image):
    landmarks = []
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if is_right_hand(handedness):
                wrist = hand_landmarks.landmark[0]  # Điểm cổ tay (landmark số 0)
                h, w, c = image.shape
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                hand_landmark_coords = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    x, y = int(lm.x * w) - wrist_x, int(lm.y * h) - wrist_y
                    hand_landmark_coords.append((x, y))
                landmarks.append(hand_landmark_coords)
    return landmarks

# Đường dẫn tới MLP_model.sav
# MLP_model.sav file's path
filename = 'MLP_model.sav' 
mlp = pickle.load(open(filename, 'rb'))

def number_to_letter(number):
    # Thứ tự chữ cái không bao gồm J và Z
    alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXY'
    return alphabet[number-1]


def preprocess_and_predict(landmarks, model): 
    landmarks = np.array(landmarks).astype(np.float32).flatten() 
    landmarks = landmarks.reshape(1, -1) 
    prediction = model.predict(landmarks) 
    return number_to_letter(prediction[0]) #

    
# Initialize MediaPipe components
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Desired aspect ratio for the bounding box (height/width)
DESIRED_ASPECT_RATIO = 3/4 # Example: 1:1 ratio for a square bounding box
PADDING = 15  # Padding for the bounding box

# Threshold for detecting hand stillness
STILLNESS_THRESHOLD = 5  # Pixel movement threshold to consider hand as "not moving"


# Function to process each video frame and detect hands
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return hands.process(rgb_frame)


# Function to calculate the bounding box from hand landmarks
def calculate_bounding_box(hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0

    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x, x_max)
        y_max = max(y, y_max)

    # Add padding to the bounding box
    x_min = max(0, x_min - PADDING)
    y_min = max(0, y_min - PADDING)
    x_max = min(w, x_max + PADDING)
    y_max = min(h, y_max + PADDING)

    return x_min, y_min, x_max, y_max


# Function to enforce a fixed aspect ratio for the bounding box
def enforce_aspect_ratio(x_min, y_min, x_max, y_max, frame_shape, desired_aspect_ratio):
    h, w, _ = frame_shape
    box_width = x_max - x_min
    box_height = y_max - y_min
    current_aspect_ratio = box_height / box_width

    if current_aspect_ratio < desired_aspect_ratio:
        # Height is too small, adjust the height
        new_height = int(box_width * desired_aspect_ratio)
        y_center = (y_min + y_max) // 2
        y_min = max(0, y_center - new_height // 2)
        y_max = min(h, y_center + new_height // 2)
    elif current_aspect_ratio > desired_aspect_ratio:
        # Width is too small, adjust the width
        new_width = int(box_height / desired_aspect_ratio)
        x_center = (x_min + x_max) // 2
        x_min = max(0, x_center - new_width // 2)
        x_max = min(w, x_center + new_width // 2)

    return x_min, y_min, x_max, y_max


# Function to draw the bounding box and landmarks on the frame
def draw_bounding_box_and_landmarks(frame, x_min, y_min, x_max, y_max, hand_landmarks):
    # Draw the bounding box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Draw landmarks on the frame
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


# Function to crop the hand region from the frame
def crop_hand(frame, x_min, y_min, x_max, y_max):
    return frame[y_min:y_max, x_min:x_max]


# Function to check if the hand is the right hand
def is_right_hand(handedness):
    return handedness.classification[0].label == 'Left'  


# Function to detect if the hand is moving
def is_hand_moving(current_landmarks, previous_landmarks, threshold, frame_shape):
    if previous_landmarks is None:
        return True  # First frame, no previous landmarks to compare

    h, w, _ = frame_shape
    total_movement = 0
    num_landmarks = len(current_landmarks)

    for lm_current, lm_previous in zip(current_landmarks, previous_landmarks):
        x_current, y_current = int(lm_current.x * w), int(lm_current.y * h)
        x_previous, y_previous = int(lm_previous.x * w), int(lm_previous.y * h)

        # Calculate the Euclidean distance between the corresponding landmarks
        distance = np.sqrt((x_current - x_previous) ** 2 + (y_current - y_previous) ** 2)
        total_movement += distance

    # Average movement per landmark
    average_movement = total_movement / num_landmarks

    return average_movement > threshold  # Hand is moving if movement is greater than threshold


# Main function to handle video capture and processing
def main():
    cap = cv2.VideoCapture(0)
    # if a camera was used, change the above line to cap = cv2.VideoCapture(0) 
    previous_hand_landmarks = None  # To store the previous frame's hand landmarks

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to detect hands
        result = process_frame(frame)

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                # Detect only the right hand
                if is_right_hand(handedness):

                    # Check if the hand is moving
                    if is_hand_moving(hand_landmarks.landmark, previous_hand_landmarks, STILLNESS_THRESHOLD, frame.shape):
                        print("Hand is moving")
                        # Run the classification
                        
                    else:
                        # Lấy tọa độ hand landmarks và in ra màn hình

                        landmarks = get_hand_landmarks(frame)
                        if landmarks :
                                 # Ensure full landmarks are captured # Chuyển đổi hand landmarks thành input cho mô hình và dự đoán 
                                prediction = preprocess_and_predict(landmarks[0], mlp) 
                                print(f"Prediction: {prediction}") 
                                cv2.putText(frame, f"Prediction: {prediction}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                    # Calculate the bounding box
                    x_min, y_min, x_max, y_max = calculate_bounding_box(hand_landmarks, frame.shape)

                    # Enforce fixed aspect ratio for the bounding box
                    x_min, y_min, x_max, y_max = enforce_aspect_ratio(
                        x_min, y_min, x_max, y_max, frame.shape, DESIRED_ASPECT_RATIO
                    )

                    # Crop the hand region
                    #hand_crop = crop_hand(frame, x_min, y_min, x_max, y_max)

                    # Show the cropped hand
                    #cv2.imshow("Cropped Right Hand", hand_crop)
                    
                    #Draw bounding box and landmarks on the original frame
                    draw_bounding_box_and_landmarks(frame, x_min, y_min, x_max, y_max, hand_landmarks)

                    # Update the previous landmarks for the next frame
                    previous_hand_landmarks = hand_landmarks.landmark

        # Display the frame
        cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()