import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def drawAnnotation(image, results):
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    return image

# distaance between two points, p1 = x1, y1
def distanceFromPalm(p1,p2):

    distance = math.sqrt(((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2))

    return distance

def fingersUp(results, image):
    fingers = []


    y, x, _ = image.shape
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            landmarkcoord = []

            for i in hand.landmark:
                landmarkcoord.append((y*i.y, x*i.x))

            fingerTipIds = [4, 8 ,12, 16, 20]

            factor = distanceFromPalm(landmarkcoord[0], landmarkcoord[1])

            for i in range(5):
                if distanceFromPalm(landmarkcoord[0], landmarkcoord[fingerTipIds[i]]) > 3 * factor:
                    fingers.append(1)
                else:
                    fingers.append(0)

    return fingers

if __name__ == '__main__':
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#            image = drawAnnotation(image, results)

            fingers = fingersUp(results, image)

#            cv2.putText(image, fingers.__str__() , (10,450) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
            cv2.putText(image, sum(fingers).__str__() , (10,150) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()