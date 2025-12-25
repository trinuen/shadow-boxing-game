import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.hands import HandLandmark

#this only works with older version of mediapipe 0.10.9
#you also need python <=3.10 to run mediapipe
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,   # disables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

direction = ["left", "right", "up", "down"]

while True:
  success, frame = cap.read()
  text = "None"

  if success:
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hand.process(RGB_frame)
    result_face = face_mesh.process(RGB_frame)

  #hands
  if result_hands.multi_hand_landmarks:
    #not using hand_idx
    for hand_idx, hand_landmarks in enumerate(result_hands.multi_hand_landmarks):

        index_mcp = hand_landmarks.landmark[5]
        index_tip = hand_landmarks.landmark[8]

        height_diff = abs(index_mcp.y - index_tip.y)
        width_diff = abs(index_mcp.x - index_tip.x)

        if height_diff <= 0.2 and index_tip.x < index_mcp.x:
          text = "pointing left"
        elif height_diff <= 0.2 and index_tip.x > index_mcp.x:
          text = "pointing right"
        elif width_diff <= 0.3 and index_tip.y < index_mcp.y:
          text = "pointing up"
        elif width_diff <= 0.3 and index_tip.y > index_mcp.y:
          text = "pointing down"
        
        wrist = hand_landmarks.landmark[0]

        # if wrist.x <= 0.3 and wrist.y > 0.3 and wrist.y < 0.7:
        #   text = direction[0] #left
        # elif wrist.x >= 0.7 and wrist.y > 0.3 and wrist.y < 0.7:
        #   text = direction[1] #right
        # elif wrist.y <= 0.3:
        #   text = direction[2] #up
        # elif wrist.y >= 0.7:
        #   text = direction[3] #down

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    height, width, _ = frame.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    #hand direction text
    x = (width - text_width) // 2
    y = text_height + 10
    cv2.putText(
      frame,
      text, 
      (x, y), 
      font, 
      font_scale, 
      (255, 255, 255), 
      thickness
    )



  #face
  if result_face.multi_face_landmarks:
    for face_landmarks in result_face.multi_face_landmarks:

      top_forehead = face_landmarks.landmark[10]

      middle_forehead = face_landmarks.landmark[151]
      middle_eyebrow = face_landmarks.landmark[9]
      low_eyebrow = face_landmarks.landmark[8]

      face_height_diff = abs(top_forehead.y - middle_eyebrow.y)
      if face_height_diff <= 0.05:
          text = "looking up"
        
      # for lm_idx, lm in enumerate(face_landmarks.landmark):
      #     print(f"LANDMARK_IDX={lm_idx} x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")

      mp_drawing.draw_landmarks(
          image=frame,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style()
      )
    #face direction text
    height, width, _ = frame.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    #hand direction text
    x = (width - text_width) // 2
    y = text_height + 10
    cv2.putText(
      frame,
      text, 
      (x, y + 10), 
      font, 
      font_scale, 
      (255, 255, 255), 
      thickness
    )

  cv2.imshow('Webcam', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()