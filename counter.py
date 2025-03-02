import cv2
import numpy as np
import mediapipe as mp
import openpyxl
import os
import time  # Added for cooldown between reps

# Load or create an Excel workbook
file_name = 'exercise_data.xlsx'
if os.path.exists(file_name):
    workbook = openpyxl.load_workbook(file_name)
    worksheet = workbook.active
else:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append(['Exercise', 'Reps', 'Sets'])

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Fixed: Initialize camera properly and handle potential failures
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Failed to open camera")
except Exception as e:
    print(f"Camera error: {e}")
    exit(1)

def calculate_angle(a, b, c):
    # Convert landmarks to numpy arrays for angle calculation
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    # Use epsilon to avoid division by zero
    epsilon = 1e-8
    cosine_angle = np.dot(ba, bc) / (max(np.linalg.norm(ba), epsilon) * max(np.linalg.norm(bc), epsilon))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def track_exercise(name, key_points, up_angle, down_angle):
    count, sets = 0, 0
    up_position = False
    feedback = f"Ready for {name}"
    
    # Added: State management variables
    last_rep_time = time.time()
    rep_cooldown = 0.5  # Seconds to prevent counting repeated reps
    consecutive_frames_threshold = 3
    consecutive_up_frames = 0
    consecutive_down_frames = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Flip frame horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = pose.process(frame_rgb)
            
            # Draw landmarks and connections
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                
                # Get landmarks
                landmarks = results.pose_landmarks.landmark
                
                # Check if all required landmarks are visible
                if all(0 <= landmarks[i].visibility <= 1 for i in key_points):
                    # Calculate angle
                    angle = calculate_angle(landmarks[key_points[0]], landmarks[key_points[1]], landmarks[key_points[2]])
                    
                    # Display angle
                    cv2.putText(frame, f"Angle: {int(angle)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # State logic with consecutive frame check for robustness
                    current_time = time.time()
                    
                    if angle > up_angle:
                        consecutive_up_frames += 1
                        consecutive_down_frames = 0
                        if consecutive_up_frames >= consecutive_frames_threshold and not up_position:
                            up_position = True
                            feedback = f"{name}: Move down"
                    
                    elif angle < down_angle:
                        consecutive_down_frames += 1
                        consecutive_up_frames = 0
                        if consecutive_down_frames >= consecutive_frames_threshold and up_position:
                            if current_time - last_rep_time > rep_cooldown:
                                count += 1
                                last_rep_time = current_time
                                up_position = False
                                feedback = f"{name} Count: {count}"
                                
                                # Set counter logic
                                if count > 0 and count % 10 == 0:
                                    sets += 1
                                    feedback = f"{name} Completed Set: {sets} (Total Reps: {count})"
                    else:
                        # Reset consecutive frame counters if in between angles
                        consecutive_up_frames = 0
                        consecutive_down_frames = 0
                else:
                    feedback = "Position not fully visible"
            else:
                feedback = "No pose detected"
            
            # Draw feedback and rep/set counters
            cv2.putText(frame, feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Reps: {count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Sets: {sets}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Display form guidance based on angle
            if results.pose_landmarks:
                if angle > up_angle - 10 and angle < up_angle + 10:
                    cv2.putText(frame, "Good starting position", (10, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                elif angle < down_angle + 10 and angle > down_angle - 10:
                    cv2.putText(frame, "Good end position", (10, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Press 'q' to quit
            cv2.imshow(f'{name} Tracker', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset counter
                count = 0
                sets = 0
                feedback = f"Reset {name} counter"
    
    return count, sets

# Main program loop
def main():
    exercise_map = {
        '1': {'name': 'Bicep Curl', 
              'landmarks': [mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                           mp_pose.PoseLandmark.LEFT_ELBOW.value, 
                           mp_pose.PoseLandmark.LEFT_WRIST.value], 
              'up_angle': 150, 'down_angle': 60},
        '2': {'name': 'Squats', 
              'landmarks': [mp_pose.PoseLandmark.LEFT_HIP.value, 
                           mp_pose.PoseLandmark.LEFT_KNEE.value, 
                           mp_pose.PoseLandmark.LEFT_ANKLE.value], 
              'up_angle': 170, 'down_angle': 90},
        '3': {'name': 'Push-Up', 
              'landmarks': [mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                           mp_pose.PoseLandmark.LEFT_ELBOW.value, 
                           mp_pose.PoseLandmark.LEFT_WRIST.value], 
              'up_angle': 160, 'down_angle': 90},
        '4': {'name': 'Shoulder Press', 
              'landmarks': [mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                           mp_pose.PoseLandmark.LEFT_ELBOW.value, 
                           mp_pose.PoseLandmark.LEFT_WRIST.value], 
              'up_angle': 160, 'down_angle': 70}
    }
    
    while True:
        print("\nSelect an exercise:")
        print("1: Bicep Curl\n2: Squat\n3: Push-Up\n4: Shoulder Press\n5: Exit")
        ch = input("Enter your choice: ")
        
        if ch == '5':
            print("Exiting...")
            break
            
        if ch not in exercise_map:
            print("Invalid choice! Try again.")
            continue
        
        exercise = exercise_map[ch]
        
        print(f"Starting {exercise['name']}...")
        print("Press 'q' to finish this exercise or 'r' to reset counter.")
        
        reps, sets = track_exercise(
            exercise['name'], 
            exercise['landmarks'], 
            exercise['up_angle'], 
            exercise['down_angle']
        )
        
        worksheet.append([exercise['name'], reps, sets])
        workbook.save(file_name)
        print(f"Recorded: {exercise['name']} - Reps: {reps}, Sets: {sets}")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure proper cleanup
        cap.release()
        cv2.destroyAllWindows()
        if 'workbook' in locals():
            workbook.close()