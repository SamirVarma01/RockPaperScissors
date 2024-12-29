import cv2
import torch
import torchvision.transforms as transforms
from model import RockPaperScissorsCNN
import mediapipe as mp

model_path = "/Applications/VSCode/MLProjs/rps/model/rock_paper_scissors_model.pth"
model = RockPaperScissorsCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

classes = ["Paper", "Rock", "Scissors"]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def predict(image, model):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)  
    h, w, c = frame.shape  
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    prediction = None

    if results.multi_hand_landmarks:  
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            roi = frame[y_min:y_max, x_min:x_max]
            try:
                prediction = predict(roi, model)
            except Exception as e:
                prediction = None

            if prediction:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"Prediction: {prediction}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Rock Paper Scissors - Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()