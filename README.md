# Visionary Assistant  
An Assistive Object Detection and Navigation System for the Visually Impaired

---

## 📌 Project Overview

Visionary Assistant is a real-time object detection and OCR-based navigation system designed to support visually impaired individuals.  
The system uses:

- **YOLOv8** for object detection  
- **PyTesseract** for reading text from the environment  
- **Ultrasonic Sensors** for distance estimation  
- **Text-to-Speech (TTS)** feedback for user alerts  
- **Camera input (Webcam or Pi Cam)** for real-time processing  

This helps users understand their surrounding environment and navigate safely with voice-guided instructions.

---

## 🚀 Features

✔️ Real-time object detection  
✔️ Text reading (OCR) using camera feed  
✔️ Distance measurement using ultrasonic sensor  
✔️ Supports **Webcam** and **Raspberry Pi Camera**  
✔️ Audio feedback using TTS  
✔️ Lightweight and optimized for Raspberry Pi  

---

## 🗂 Repository Structure

| File Name | Description |
|----------|------------|
| `objectdetection(webcam).py` | Runs YOLO object detection using a USB/Webcam |
| `objectdetection(picam).py` | Runs YOLO object detection using Raspberry Pi Camera |
| `charactordetection(webcam).py` | OCR (PyTesseract) text detection using Webcam |
| `charactordetection(picam).py` | OCR (PyTesseract) text detection using Pi Camera |
| `ultrasonic.py` | Code for reading distance values using HC-SR04 ultrasonic sensor |
| `README.md` | Documentation file (you are reading it now 🙂) |

---

## 🛠️ Requirements

### Hardware
- Raspberry Pi 4 / Laptop  
- Pi Camera or USB Webcam  
- HC-SR04 Ultrasonic Sensor  
- Speaker / Headphones  

### Software & Libraries

Install dependencies:

pip install ultralytics opencv-python pytesseract pyttsx3

sudo apt-get install tesseract-ocr




## 🔊 Output Format

The system provides real-time voice feedback, including object names, distance, and detected text.

Example responses:

- **"Person detected at 92 centimeters."**
- **"Text detected: Welcome to Station."**
- **"Bottle detected ahead, 57 centimeters."**
- **"No text found."**


---

## 🧪 Applications

- 👨‍🦯 Guide system for visually impaired individuals  
- 🧭 Smart navigation and obstacle awareness  
- 🔍 Real-time environment detection  
- 📚 Text reading assistance in public places  
- 🤖 AI-powered mobility and accessibility tools  


---

## 🚀 Future Improvements

- 📍 GPS-assisted outdoor navigation  
- ☁️ Cloud-based analytics and remote monitoring  
- ⚡ TensorRT optimization for faster inference  
- 🌍 Multi-language speech output  
- 🔧 Edge TPU / NVIDIA Jetson Nano acceleration  
- 🔋 Improved power efficiency and modular hardware  


---

## 👨‍🎓 Developer

| Name | Role |
|------|------|
| **Adesh Kushwaha** | Developer & System Engineer |

---



