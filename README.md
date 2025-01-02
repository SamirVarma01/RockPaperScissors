# Rock, Paper, Scissors Detection Project

## Overview
This project uses a Convolutional Neural Network (CNN) to classify hand gestures representing **Rock**, **Paper**, or **Scissors**. It integrates a web application built using Flask for user interaction and real-time gesture recognition using OpenCV.

The application detects gestures through a webcam, classifies them, and draws a bounding box around the detected hand.

## Usage

To use the app, first clone the repository. I could not get some elements to load in other files without using their absolute path, so in some cases you may see the pathname on my computer. Please rephrase these to fit your directory. Afterwards, start a terminal in the directory and use python app.py. Once this is done, you can visit the locally hosted game at http://127.0.0.1:5000/. 

NOTE: Scissors is a little difficult to detect for the model. The best way I have found is using your left hand to point two finger straight upwards, as if pointing a finger gun in the air.
---

## Features
- Image Classification: Classify static images into Rock, Paper, or Scissors.

- Web Application: Upload images and get classification results using a Flask-based interface.

- Real-Time Gesture Recognition: Use a webcam to detect and classify hand gestures in real time.

- Interactive Feedback: Bounding boxes and labels are drawn on the webcam feed for real-time feedback.

## Future Enhancements
- Improved Hand Detection: Use advanced hand segmentation or detection methods for better accuracy. In particular, there is some issues in detection for scissors, so this could be improved with a better dataset.

- Multiplayer Game: Extend the application to support multiplayer rock-paper-scissors games using webcams.

- Mobile Integration: Deploy the model on mobile devices for gesture recognition on the go.

# Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests for bug fixes, enhancements, or new features.



