from flask import Flask, render_template, Response, jsonify, request
import cv2
import torch
import torchvision.transforms as transforms
from model import RockPaperScissorsCNN
import random
import time
import mediapipe as mp

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

model_path = "/Applications/VSCode/MLProjs/rps/model/rock_paper_scissors_model.pth"
model = RockPaperScissorsCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

classes = ["Paper", "Rock", "Scissors"]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict(image, model):
    image = transform(image).unsqueeze(0)  
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

@app.route('/')
def index():
    return render_template('index.html')

def initialize_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap

camera = initialize_camera()

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global camera
        while True:
            if not camera.isOpened():
                camera = initialize_camera()
                if not camera.isOpened():
                    print("Failed to reinitialize camera")
                    time.sleep(0.1)
                    continue

            success, frame = camera.read()
            if not success:
                print("Failed to capture frame")
                camera.release()
                camera = initialize_camera()
                time.sleep(0.1)
                continue

            try:
                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
                        y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
                        x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
                        y_max = max([lm.y for lm in hand_landmarks.landmark]) * h
                        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                        roi = frame[y_min:y_max, x_min:x_max]
                        if roi.size != 0:  
                            try:
                                prediction = predict(roi, model)
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                cv2.putText(frame, f"Prediction: {prediction}", 
                                          (x_min, y_min - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 
                                          cv2.LINE_AA)
                            except Exception as e:
                                print(f"Prediction error: {e}")

                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in frame processing: {e}")
                continue

    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/play', methods=['POST'])
def play():
    global camera
    countdown = 3
    for i in range(countdown, 0, -1):
        print(f"Countdown: {i}")  
        time.sleep(1)

    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Failed to capture frame. Try again.'})

    x_start, y_start, x_end, y_end = 100, 100, 400, 400
    roi = frame[y_start:y_end, x_start:x_end]  
    prediction = predict(roi, model)

    bot_choice = random.choice(classes)

    if prediction == bot_choice:
        result = "It's a tie!"
    elif (prediction == "Rock" and bot_choice == "Scissors") or \
         (prediction == "Scissors" and bot_choice == "Paper") or \
         (prediction == "Paper" and bot_choice == "Rock"):
        result = "You win!"
    else:
        result = "Bot wins!"

    return jsonify({'player_choice': prediction, 'bot_choice': bot_choice, 'result': result})

@app.route('/stop')
def stop():
    cap.release()
    cv2.destroyAllWindows()
    return "Webcam released and app stopped."

@app.route('/test_camera')
def test_camera():
    success, frame = camera.read()
    if not success:
        return "Failed to capture frame", 500
    
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return "Failed to encode frame", 500
    
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.teardown_appcontext
def teardown_camera(exception):
    global camera
    camera.release()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)