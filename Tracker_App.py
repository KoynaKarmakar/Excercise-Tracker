import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd
import os
from datetime import datetime

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Set page config
st.set_page_config(
    page_title="Exercise Rep Counter",
    page_icon="💪",
    layout="wide"
)

# Calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    # Avoid division by zero
    epsilon = 1e-8
    cosine_angle = np.dot(ba, bc) / (max(np.linalg.norm(ba), epsilon) * max(np.linalg.norm(bc), epsilon))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return angle

# Initialize exercise configurations
exercise_configs = {
    'Bicep Curl': {
        'landmarks': [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value
        ],
        'up_angle': 150,
        'down_angle': 60,
        'description': 'Bend your elbow to bring the weight up toward your shoulder'
    },
    'Squats': {
        'landmarks': [
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.LEFT_ANKLE.value
        ],
        'up_angle': 170,
        'down_angle': 90,
        'description': 'Bend your knees and lower your body as if sitting in a chair'
    },
    'Push-Up': {
        'landmarks': [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value
        ],
        'up_angle': 160,
        'down_angle': 90,
        'description': 'Lower your body by bending your elbows, then push back up'
    },
    'Shoulder Press': {
        'landmarks': [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value
        ],
        'up_angle': 160,
        'down_angle': 70,
        'description': 'Raise weights above your head by straightening your arms'
    }
}

# App title and description
st.title("💪 Exercise Rep Counter")
st.markdown("Track your exercise repetitions and sets using computer vision!")

# Create sidebar for controls
st.sidebar.header("Controls")

# Exercise selection
selected_exercise = st.sidebar.selectbox(
    "Select Exercise",
    list(exercise_configs.keys())
)

# Advanced settings expander
with st.sidebar.expander("Advanced Settings"):
    # Allow users to customize angle thresholds
    up_angle = st.slider(
        "Up Position Angle", 
        min_value=120, 
        max_value=180, 
        value=exercise_configs[selected_exercise]['up_angle'],
        help="Angle threshold for the up position"
    )
    
    down_angle = st.slider(
        "Down Position Angle", 
        min_value=30, 
        max_value=120, 
        value=exercise_configs[selected_exercise]['down_angle'],
        help="Angle threshold for the down position"
    )
    
    # Rep detection sensitivity
    consecutive_frames = st.slider(
        "Detection Smoothing", 
        min_value=1, 
        max_value=5, 
        value=3,
        help="Higher values make detection more stable but less responsive"
    )
    
    rep_cooldown = st.slider(
        "Rep Cooldown (seconds)", 
        min_value=0.1, 
        max_value=2.0, 
        value=0.5, 
        step=0.1,
        help="Minimum time between rep counts"
    )

# Update exercise config with user settings
exercise_configs[selected_exercise]['up_angle'] = up_angle
exercise_configs[selected_exercise]['down_angle'] = down_angle

# Workout statistics
st.sidebar.header("Workout Statistics")
if 'total_reps' not in st.session_state:
    st.session_state.total_reps = 0
if 'sets_completed' not in st.session_state:
    st.session_state.sets_completed = 0
if 'workout_log' not in st.session_state:
    st.session_state.workout_log = []

st.sidebar.metric("Total Reps", st.session_state.total_reps)
st.sidebar.metric("Sets Completed", st.session_state.sets_completed)

# Reset button
if st.sidebar.button("Reset Counters"):
    st.session_state.current_reps = 0
    st.session_state.current_sets = 0

# Save workout button
if st.sidebar.button("Save Workout"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    workout_entry = {
        'timestamp': now,
        'exercise': selected_exercise,
        'reps': st.session_state.total_reps,
        'sets': st.session_state.sets_completed
    }
    st.session_state.workout_log.append(workout_entry)
    st.sidebar.success("Workout saved!")
    
    # Save to CSV
    df = pd.DataFrame(st.session_state.workout_log)
    df.to_csv('workout_history.csv', index=False)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Create a placeholder for the webcam feed
    frame_placeholder = st.empty()
    
    # Create placeholders for metrics
    metrics_placeholder = st.empty()
    
with col2:
    # Exercise info card
    st.subheader(f"{selected_exercise}")
    st.write(exercise_configs[selected_exercise]['description'])
    st.markdown("---")
    
    # Current exercise stats
    exercise_stats = st.empty()
    
    # Feedback area
    feedback_placeholder = st.empty()

# Start button
start_button = st.button("Start Exercise Tracking")

if start_button:
    # Initialize counters for current session
    if 'current_reps' not in st.session_state:
        st.session_state.current_reps = 0
    if 'current_sets' not in st.session_state:
        st.session_state.current_sets = 0
    
    # Get exercise config
    config = exercise_configs[selected_exercise]
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Get key points for the selected exercise
    key_points = config['landmarks']
    
    # Stop button
    stop_button = st.button("Stop Tracking", key=f"stop_button_{time.time()}")

    # Initialize pose detection
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        # State variables
        up_position = False
        feedback = f"Ready for {selected_exercise}"
        last_rep_time = time.time()
        consecutive_up_frames = 0
        consecutive_down_frames = 0
        
        # Main loop
        try:
            while cap.isOpened():
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video")
                    break
                
                # Flip horizontally for mirror view
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the image for pose detection
                results = pose.process(frame_rgb)
                
                # Draw landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_rgb, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )
                    
                    # Get landmarks
                    landmarks = results.pose_landmarks.landmark
                    
                    # Check if landmarks are visible
                    if all(0 <= landmarks[i].visibility <= 1 for i in key_points):
                        # Calculate angle
                        angle = calculate_angle(
                            landmarks[key_points[0]],
                            landmarks[key_points[1]],
                            landmarks[key_points[2]]
                        )
                        
                        # Draw angle on frame
                        cv2.putText(
                            frame_rgb,
                            f"Angle: {int(angle)}", 
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA
                        )
                        
                        # State logic with consecutive frame check
                        current_time = time.time()
                        
                        # Check for up position
                        if angle > config['up_angle']:
                            consecutive_up_frames += 1
                            consecutive_down_frames = 0
                            if consecutive_up_frames >= consecutive_frames and not up_position:
                                up_position = True
                                feedback = f"{selected_exercise}: Move down"
                        
                        # Check for down position
                        elif angle < config['down_angle']:
                            consecutive_down_frames += 1
                            consecutive_up_frames = 0
                            if consecutive_down_frames >= consecutive_frames and up_position:
                                if current_time - last_rep_time > rep_cooldown:
                                    st.session_state.current_reps += 1
                                    st.session_state.total_reps += 1
                                    last_rep_time = current_time
                                    up_position = False
                                    feedback = f"Good rep! Count: {st.session_state.current_reps}"
                                    
                                    # Set counter logic
                                    if st.session_state.current_reps > 0 and st.session_state.current_reps % 10 == 0:
                                        st.session_state.current_sets += 1
                                        st.session_state.sets_completed += 1
                                        feedback = f"Completed Set: {st.session_state.current_sets}!"
                        
                        else:
                            # Reset consecutive frame counters if in between angles
                            consecutive_up_frames = 0
                            consecutive_down_frames = 0
                        
                        # Form guidance
                        if angle > config['up_angle'] - 10 and angle < config['up_angle'] + 10:
                            form_feedback = "Good starting position"
                        elif angle < config['down_angle'] + 10 and angle > config['down_angle'] - 10:
                            form_feedback = "Good end position"
                        else:
                            form_feedback = "Move to complete the rep"
                        
                        # Display form guidance
                        cv2.putText(
                            frame_rgb,
                            form_feedback,
                            (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )
                    else:
                        feedback = "Position not fully visible"
                else:
                    feedback = "No pose detected"
                
                # Update metrics in the sidebar
                with metrics_placeholder.container():
                    metrics_cols = st.columns(3)
                    metrics_cols[0].metric("Current Angle", int(angle) if 'angle' in locals() else 0)
                    metrics_cols[1].metric("Reps", st.session_state.current_reps)
                    metrics_cols[2].metric("Sets", st.session_state.current_sets)
                
                # Update exercise stats
                exercise_stats.markdown(f"""
                **Current Exercise:** {selected_exercise}  
                **Reps in current set:** {st.session_state.current_reps % 10 if st.session_state.current_reps > 0 else 0}/10  
                **Target angles:** {config['down_angle']}° (down) to {config['up_angle']}° (up)
                """)
                
                # Update feedback
                feedback_placeholder.info(feedback)
                
                # Display the frame
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Check for stop button (implemented as a checkbox for continuous checking)
                
                #stop_button_pressed = stop_placeholder.button("Stop Tracking", key=f"stop_button_{time.time()}")
                if stop_button:
                    break
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            # Release resources
            cap.release()

# Display workout history
if st.session_state.workout_log:
    st.markdown("---")
    st.subheader("Workout History")
    workout_df = pd.DataFrame(st.session_state.workout_log)
    st.dataframe(workout_df)
    
    # Download button for workout history
    csv = workout_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Workout History",
        csv,
        "workout_history.csv",
        "text/csv",
        key='download-csv'
    )
