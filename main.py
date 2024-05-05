import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.hotkey('ctrl', 't')  # Open a new tab
time.sleep(1)
pyautogui.write('chrome://dino', interval=0.1)
pyautogui.press('enter')
time.sleep(2)

    # Set focus on the game window (adjust coordinates accordingly)
pyautogui.click(x=500, y=500)
time.sleep(1)
mp_drawing = mp.solutions.drawing_utils
mphands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mphands.Hands()

# Initialize previous fingertip positions
prev_fingertip_positions = [None, None]

# Initialize space bar action variables
space_bar_delay = 0.5  # Adjust the delay as needed (in seconds)
space_bar_last_pressed_time = time.time()

while True:
    data, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the positions of the index and middle fingertips (landmark #8 and #12)
            index_fingertip = (
                int(hand_landmarks.landmark[8].x * image.shape[1]),
                int(hand_landmarks.landmark[8].y * image.shape[0])
            )
            thumb_fingertip = (
                int(hand_landmarks.landmark[4].x * image.shape[1]),
                int(hand_landmarks.landmark[4].y * image.shape[0])
            )

            # Draw circles at the current fingertip positions
            cv2.circle(image, index_fingertip, 10, (0, 255, 0), -1)
            cv2.circle(image, thumb_fingertip, 10, (0, 255, 0), -1)

            # Check if the index and middle fingertips are close together
            if all(prev_fingertip_positions) and \
                    cv2.norm(index_fingertip, thumb_fingertip) < 50:
                # Check if space bar should be pressed
                if time.time() - space_bar_last_pressed_time > space_bar_delay:
                    pyautogui.press('space')
                    space_bar_last_pressed_time = time.time()

            # Update the previous fingertip positions
            prev_fingertip_positions = [index_fingertip, thumb_fingertip]

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, hand_landmarks, mphands.HAND_CONNECTIONS)

    cv2.imshow('Handtracker', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
