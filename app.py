import cv2
import mediapipe as mp

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
HAND_LANDMARKS = mp.solutions.hands.HandLandmark

# --- GESTURE DETECTION FUNCTION: DETECTS CLENCHED FIST WITH HIGH TOLERANCE AND SIDE-POSE FILTER ---

# Adjusted Tolerances
CLENCH_TOLERANCE = 0.05 # Reduced slightly for better discrimination
MCP_TOLERANCE = 0.04  # Stricter check against main knuckles to prevent 'cupping' detection
SPLAY_THRESHOLD = 0.10 # Max allowed horizontal difference between index and pinky knuckle (Normalized to 0.0-1.0)

def is_clenched_fist(hand_landmarks):
    """
    Checks for a clenched fist, using tolerance and robust checks, 
    including a filter to reject hands viewed from the side (like a 'Done' sign).
    """
    
    # Landmark indices for the four fingers (Tip, PIP, and MCP)
    finger_data = [
        (HAND_LANDMARKS.INDEX_FINGER_TIP, HAND_LANDMARKS.INDEX_FINGER_PIP, HAND_LANDMARKS.INDEX_FINGER_MCP),
        (HAND_LANDMARKS.MIDDLE_FINGER_TIP, HAND_LANDMARKS.MIDDLE_FINGER_PIP, HAND_LANDMARKS.MIDDLE_FINGER_MCP),
        (HAND_LANDMARKS.RING_FINGER_TIP, HAND_LANDMARKS.RING_FINGER_PIP, HAND_LANDMARKS.RING_FINGER_MCP),
        (HAND_LANDMARKS.PINKY_TIP, HAND_LANDMARKS.PINKY_PIP, HAND_LANDMARKS.PINKY_MCP), 
    ]
    
    is_clenched = True
    
    # ----------------------------------------------------
    # NEW: 1. KNUCKLE SPLAY CHECK (Eliminates side poses)
    # ----------------------------------------------------
    index_mcp_x = hand_landmarks.landmark[HAND_LANDMARKS.INDEX_FINGER_MCP].x
    pinky_mcp_x = hand_landmarks.landmark[HAND_LANDMARKS.PINKY_MCP].x
    
    # If the horizontal distance between the index and pinky knuckles is large, 
    # the hand is either open or viewed from the side, so we reject it.
    knuckle_spread = abs(index_mcp_x - pinky_mcp_x)
    if knuckle_spread > SPLAY_THRESHOLD:
        return False # Definitely not a front-facing clenched fist

    # ----------------------------------------------------
    # 2. FINGER BENDING CHECK (Clenched Finger Logic)
    # ----------------------------------------------------
    for tip_lm, pip_lm, mcp_lm in finger_data:
        tip_y = hand_landmarks.landmark[tip_lm].y
        pip_y = hand_landmarks.landmark[pip_lm].y
        mcp_y = hand_landmarks.landmark[mcp_lm].y

        # Condition 1 (PIP Check): Tip must be below the middle joint (PIP) minus tolerance
        pip_check = tip_y > (pip_y - CLENCH_TOLERANCE)
        
        # Condition 2 (MCP Check): Tip must be clearly below the main knuckle (MCP) + tolerance
        mcp_check = tip_y > (mcp_y + MCP_TOLERANCE) 
        
        if not pip_check or not mcp_check:
            is_clenched = False
            break 
            
    # ----------------------------------------------------
    # 3. THUMB CHECK
    # ----------------------------------------------------
    thumb_tip_y = hand_landmarks.landmark[HAND_LANDMARKS.THUMB_TIP].y
    thumb_ip_y = hand_landmarks.landmark[HAND_LANDMARKS.THUMB_IP].y
    
    is_thumb_closed = (thumb_tip_y > (thumb_ip_y - CLENCH_TOLERANCE))
    
    # Final result: Fingers are bent AND thumb is bent AND hand is not viewed from the side.
    return is_clenched and is_thumb_closed

# ------------------------------------

# Start the video capture
cap = cv2.VideoCapture(0)

# Start MediaPipe Hands processing 
with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Pre-process the image
        image = cv2.flip(image, 1) 
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process the image
        results = hands.process(image_rgb)

        # Post-process the image for display
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        detection_status = "Scanning for Help Sign (Clenched Fist)..."

        # Draw hand landmarks and check for the gesture
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Check for the 'Help' gesture (Clenched Fist)
                if is_clenched_fist(hand_landmarks):
                    detection_status = "!!! HELP SIGN DETECTED !!!"
                    
                    # Draw a big RED box when the gesture is detected
                    h, w, c = image.shape
                    cv2.rectangle(image, (w//4, h//4), (w*3//4, h*3//4), (0, 0, 255), 10) 
                
                # Draw the hand skeleton and landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), 
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2) 
                )

        # Display the detection status text on the image
        cv2.putText(image, detection_status, (20, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

        # Display the final frame
        cv2.imshow('MediaPipe Hand Detection Project', image)

        # Press 'q' or 'Q' to quit the window
        if cv2.waitKey(5) & 0xFF in [ord('q'), ord('Q')]:
            break

# 4. Cleanup and release resources
cap.release()
cv2.destroyAllWindows()