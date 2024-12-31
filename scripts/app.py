from flask import Flask, render_template, Response, jsonify, request
import cv2
import torch
import torchvision.transforms as transforms
from model import RockPaperScissorsCNN
import random
import time

app = Flask(__name__)

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

cap = cv2.VideoCapture(0)

def predict(image, model):
    image = transform(image).unsqueeze(0)  
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/play', methods=['POST'])
def play():
    countdown = 3
    for i in range(countdown, 0, -1):
        print(f"Countdown: {i}")  
        time.sleep(1)

    success, frame = cap.read()
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

if __name__ == "__main__":
    app.run(debug=True)