import cv2
import mediapipe as mp
import numpy as np
import random
import time
from numpy.typing import NDArray

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
# Define how long the loop should run (in seconds)
DURATION = 8
won = True

def display_text(frame: NDArray[np.uint8], text: str, height: int, width: int, font_scale: int, thickness: int, color: tuple[int]):
  font = cv2.FONT_HERSHEY_SIMPLEX
  (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

  x = (width - text_width) // 2
  y = text_height + baseline + height
  
  cv2.putText(
  frame,
  text, 
  (x,y), 
  font, 
  font_scale, 
  color, 
  thickness
  )

def rest_period(your_choice: str, bot_choice: str, on_offense: bool):
  print("in rest period")
  rest_time = 8

  if your_choice == bot_choice and on_offense:
    color = (0,255,0)
    text = "GOOD HIT"
  elif your_choice == bot_choice and not on_offense:
    color = (0,0,255)
    text = "YOU GOT HIT"
  elif your_choice != bot_choice and not on_offense:
    color = (0,255,0)
    text = "SUCCESSFUL WEAVE"
  else:
    color = (0,0,255)
    text = "WRONG, BOT CHOSE: " + bot_choice
  
  start_time = time.time()
  while time.time() - start_time <= rest_time:
    #display the time:
    success, frame = cap.read()
    time_left = rest_time - round(time.time() - start_time)
    display_text(frame, str(time_left), 0, 50, 2, 2, (0,0,255))
    height, width, _ = frame.shape
    display_text(frame, text, 0, width, 1, 2, color)

    cv2.imshow('Webcam', frame)

    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


def offense():
  print("On Offense")

  global game_over
  on_offense = True
  directions = ["pointing left", "pointing right", "pointing up", "pointing down"]

  #not on offense anymore if guess the wrong direction
  while on_offense and not game_over:
    #bot choice
    random_direction = random.choice(directions)
    directions.remove(random_direction)
    print(f"bot chose: {random_direction}")

    # Record the start time
    start_time = time.time()

    while time.time() - start_time <= DURATION:
      #display the time:
      success, frame = cap.read()
      text = "None"
      time_left = DURATION - round(time.time() - start_time)
      display_text(frame, str(time_left), 4, 50, 2, 2, (0,0,255))

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
            
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        height, width, _ = frame.shape
        display_text(frame, text, 0, width, 1, 2, (255, 255, 255))

      cv2.imshow('Webcam', frame)

      time.sleep(0.1)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        game_over = True
        on_offense = False
        break
    
    print(f"Your choice: {text}")

    if random_direction != text:
      on_offense = False

    if len(directions) == 1 and random_direction == text:
      print("You Won")
      game_over = True
      break

    rest_period(text, random_direction, True)
    
def defense():
  print("On Defense")

  global game_over
  on_defense = True
  directions = ["looking left", "looking right", "looking up", "looking down"]

  #not on offense anymore if guess the wrong direction
  while on_defense and not game_over:
    #bot choice
    random_direction = random.choice(directions)
    directions.remove(random_direction)
    print(f"bot chose: {random_direction}")

    # Record the start time
    start_time = time.time()
    while time.time() - start_time <= DURATION:
      success, frame = cap.read()
      text = "None"
      time_left = DURATION - round(time.time() - start_time)
      display_text(frame, str(time_left), 0, 50, 2, 2, (255, 255, 255))

      if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_hands = hand.process(RGB_frame)
        result_face = face_mesh.process(RGB_frame)

      #face
      if result_face.multi_face_landmarks:
        for face_landmarks in result_face.multi_face_landmarks:

          #top of face
          top_forehead = face_landmarks.landmark[10]
          middle_eyebrow = face_landmarks.landmark[9]

          #bottom of face
          under_lip = face_landmarks.landmark[18]
          chin = face_landmarks.landmark[175]

          top_nose = face_landmarks.landmark[4]
          #face right side
          right_right_cheek = face_landmarks.landmark[352]

          #face left side
          left_left_cheek = face_landmarks.landmark[137]

          forehead_height_diff = abs(top_forehead.y - middle_eyebrow.y)
          chin_height_diff = abs(under_lip.y - chin.y)
          right_cheek_width_diff = abs(top_nose.x - right_right_cheek.x)
          left_cheek_width_diff = abs(top_nose.x - left_left_cheek.x)

          if forehead_height_diff <= 0.04:
              text = "looking up"
          elif chin_height_diff <= 0.035:
              text = "looking down"
          elif right_cheek_width_diff <= 0.03:
              text = "looking right"
          elif left_cheek_width_diff <= 0.03:
              text = "looking left"

          mp_drawing.draw_landmarks(
              image=frame,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style()
          )

        height, width, _ = frame.shape
        display_text(frame, text, 0, width, 1, 2, (255, 255, 255))

      cv2.imshow('Webcam', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        on_defense = False
        game_over = True
        break

    print(f"Your choice: {text}")

    if random_direction != text:
      on_defense = False

    if len(directions) == 1 and random_direction == text:
      print("You Lost")
      won = False
      game_over = True
      break

    rest_period(text, random_direction, False)

game_over = False

#game loop
while not game_over:
  offense()
  defense()

#ending screen
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    height, width, _ = frame.shape
    if won:
      display_text(frame, "You Won", 0, width, 1, 2, (255, 255, 255))
    else:
      display_text(frame, "You Lost", 0, width, 1, 2, (255, 255, 255))

    cv2.imshow("Webcam", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()