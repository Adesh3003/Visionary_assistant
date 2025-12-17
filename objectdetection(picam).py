#!/usr/bin/env python3
# YOLO zone detection + Ultrasonic distance + pico2wave TTS
# Behaviour:
#  - If distance <= 85 cm: "<names> detected at <N> centimeters far."
#  - If distance > 85 cm: "<names> detected."

from ultralytics import YOLO
import cv2
import numpy as np
import time
import subprocess
import threading
import queue
import os
import RPi.GPIO as GPIO


MODEL_WEIGHTS = "yolo11n.pt"

zone_x_min = 0.3
zone_x_max = 0.7
zone_y_min = 0.2
zone_y_max = 0.8

IMG_SIZE = 320
CONF_TH = 0.35


DETECT_EVERY_N_FRAMES = 2


PICO_LANG = "en-GB"
COOLDOWN_S = 3.0    


speak_q = queue.Queue()

def tts_worker():
    while True:
        text = speak_q.get()
        if text is None:
            break
        wav_path = "tts_out.wav"
        try:
            subprocess.run(["pico2wave", "-l", PICO_LANG, "-w", wav_path, text], check=True)
            subprocess.run(["aplay", "-q", wav_path], check=False)
        except Exception as e:
            print("[TTS ERROR]", e)
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            speak_q.task_done()

def speak(text):
    text = text.replace("_", " ")
    print("ðŸ”Š", text)
    speak_q.put(text)

threading.Thread(target=tts_worker, daemon=True).start()


def phrase_with_and(names):
    names = list(dict.fromkeys(names))  
    n = len(names)
    if n == 1:
        return names[0]
    if n == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


TRIG = 23
ECHO = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIG, False)
time.sleep(2)


last_spoken_distance = None
last_distance_time = 0

last_spoken_time = 0



rpicam_cmd = [
    "rpicam-vid",
    "--codec", "mjpeg",
    "--width", "640",
    "--height", "480",
    "--framerate", "15",
    "--nopreview",      
    "-t", "0",          
    "-o", "-"          


proc = subprocess.Popen(
    rpicam_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    bufsize=0
)

mjpeg_buffer = bytearray()

def read_frame_from_rpicam():
    """Read one JPEG frame from rpicam-vid MJPEG stream and decode to BGR image."""
    global mjpeg_buffer
    if proc.stdout is None:
        return None

    while True:
        chunk = proc.stdout.read(1024)
        if not chunk:
            return None  

        mjpeg_buffer += chunk

        soi = mjpeg_buffer.find(b'\xff\xd8')  # Start Of Image
        eoi = mjpeg_buffer.find(b'\xff\xd9')  # End Of Image

        if soi != -1 and eoi != -1 and eoi > soi:
            jpg = mjpeg_buffer[soi:eoi+2]
            mjpeg_buffer = mjpeg_buffer[eoi+2:]

            img_array = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame


# Load YOLO
model = YOLO(MODEL_WEIGHTS)

print("Running optimized rpicam-vid: YOLO zone detection + Ultrasonic distance + TTS (press 'q' to quit)")

frame_counter = 0
last_results = None  

try:
    while True:
        frame = read_frame_from_rpicam()
        if frame is None:
            print("âš  Could not read frame from rpicam-vid")
            break

        h, w, _ = frame.shape
        annotated = frame.copy()

        frame_counter += 1

        
        if frame_counter % DETECT_EVERY_N_FRAMES == 0 or last_results is None:
            last_results = model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_TH,
                device="cpu",
                verbose=False
            )

        results = last_results
        now = time.time()
        names_in_zone = []

        # draw zone rectangle
        zx1, zy1 = int(zone_x_min * w), int(zone_y_min * h)
        zx2, zy2 = int(zone_x_max * w), int(zone_y_max * h)
        cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (255, 0, 0), 2)

        
        if results and len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cls = int(box.cls[0])
                name = model.names[cls]

                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                cx_norm = cx / w
                cy_norm = cy / h

                if (zone_x_min <= cx_norm <= zone_x_max) and (zone_y_min <= cy_norm <= zone_y_max):
                    names_in_zone.append(name)
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(annotated, name, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        #  Ultrasonic distance measurement 
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()
        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = round((pulse_duration * 34300) / 2, 2)
        print(f"Distance: {distance} cm")

        # Decide what to speak 
        if names_in_zone and (now - last_spoken_time > COOLDOWN_S):
            names_phrase = phrase_with_and(names_in_zone)

            if distance <= 85:
                # Keep your ultrasonic anti-spam rule when speaking distance
                can_say_distance = False
                if (last_spoken_distance is None) or \
                   ((abs(distance - last_spoken_distance) > 2 and time.time() - last_distance_time > 2)
                    or (time.time() - last_distance_time > 10)):
                    can_say_distance = True

                if can_say_distance:
                    speak(f"{names_phrase} detected at {int(distance)} centimeters far.")
                    last_spoken_distance = distance
                    last_distance_time = time.time()
                    last_spoken_time = now  # global cooldown

            else:
                # > 85 cm â†’ say only names (no ultrasonic gating, just cooldown)
                speak(f"{names_phrase} detected.")
                last_spoken_time = now  

        #  Show window & key handling 
        cv2.imshow("YOLO Zone + Ultrasonic Distance (press q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    cv2.destroyAllWindows()
    speak_q.put(None)
    GPIO.cleanup()
    try:
        if proc:
            proc.terminate()
    except Exception:
        pass
