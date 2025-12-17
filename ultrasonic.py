
import RPi.GPIO as GPIO
import time, os, subprocess

# Pin setup 
TRIG = 23
ECHO = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIG, False)
time.sleep(2)

# TTS config
TTS_MODE = "pico"     
LANG_PICO = "en-GB"   
LANG_ESPEAK = "hi"   
last_spoken = None
last_time = 0

def speak(text):
    """Speak given text using already-installed TTS."""
    print("??", text)
    if TTS_MODE == "pico":
        os.system(f'pico2wave -l {LANG_PICO} -w tts.wav "{text}" && aplay -q tts.wav && rm -f tts.wav')
    else:
        subprocess.run(["espeak-ng", "-v", LANG_ESPEAK, text])

#  Main loop
print("Ultrasonic Sensor + TTS (Ctrl+C to stop)")

try:
    while True:
      
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        # wait for echo start
        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()
        # wait for echo end
        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()

        # compute distance
        pulse_duration = pulse_end - pulse_start
        distance = round((pulse_duration * 34300) / 2, 2)
        print(f"Distance: {distance} cm")

        # condition 
        if distance <= 85:   # speak only when = 85 cm
            
            if (last_spoken is None) or (abs(distance - last_spoken) > 2 and time.time() - last_time > 2) \
               or (time.time() - last_time > 10):
                speak(f"Distance {int(distance)} centimeters far.")
                last_spoken = distance
                last_time = time.time()

        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    GPIO.cleanup()
