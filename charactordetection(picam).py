
import cv2
import pytesseract
import subprocess
import os
import time
import sys
import numpy as np  # 

class OCRReader:
    def __init__(self):
        # Configuration
        self.cam_width = 640
        self.cam_height = 480
        self.fps = 15
        self.speak_delay = 5  # Seconds between auto-speak
        self.process_every = 2  # Process every Nth frame
        self.min_text_length = 3  # Minimum characters to speak
        self.roi_scale = 0.5  # ROI size (0.5 = middle 50%)
        
        # State variables
        self.last_spoken = ""
        self.last_speak_time = 0
        self.frame_count = 0
        self.is_speaking = False
        
        # Camera process / buffer for rpicam-vid
        self.proc = None
        self.mjpeg_buffer = bytearray()
    
    def initialize_camera(self):
        """Initialize Pi Camera using rpicam-vid MJPEG stream"""
        print("Initializing Pi Camera via rpicam-vid...")
        
        rpicam_cmd = [
            "rpicam-vid",
            "--codec", "mjpeg",
            "--width", str(self.cam_width),
            "--height", str(self.cam_height),
            "--framerate", str(self.fps),
            "--nopreview",      
            "-t", "0",         
            "-o", "-"          
        ]
        
        try:
            self.proc = subprocess.Popen(
                rpicam_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0
            )
        except FileNotFoundError:
            print("ERROR: rpicam-vid command not found!")
            print("Make sure Raspberry Pi OS (Bookworm) camera stack is installed.")
            return False
        
        # Test: read first frame
        frame = self.read_frame_from_rpicam(timeout=5.0)
        if frame is None:
            print("ERROR: Could not read initial frame from rpicam-vid.")
            return False
        
        print("Camera initialized successfully!")
        return True
    
    def read_frame_from_rpicam(self, timeout=None):
        """
        Read one JPEG frame from rpicam-vid MJPEG stream and decode to BGR image.
        If timeout is set, will stop trying after given seconds.
        """
        if self.proc is None or self.proc.stdout is None:
            return None
        
        start_time = time.time()
        
        while True:
            if timeout is not None and (time.time() - start_time) > timeout:
                return None
            
            chunk = self.proc.stdout.read(1024)
            if not chunk:
                return None  
            
            self.mjpeg_buffer += chunk
            
            soi = self.mjpeg_buffer.find(b'\xff\xd8')  # Start Of Image
            eoi = self.mjpeg_buffer.find(b'\xff\xd9')  # End Of Image
            
            if soi != -1 and eoi != -1 and eoi > soi:
                jpg = self.mjpeg_buffer[soi:eoi+2]
                self.mjpeg_buffer = self.mjpeg_buffer[eoi+2:]
                
                img_array = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return frame
    
    def speak(self, text):
        """Convert text to speech with error handling"""
        if not text.strip() or len(text) < self.min_text_length:
            return False
        
        try:
            self.is_speaking = True
            # Generate speech file
            subprocess.run(
                ['pico2wave', '-w', '/tmp/ocr_speech.wav', text],
                check=True, capture_output=True, timeout=5
            )
            # Play speech
            subprocess.run(
                ['aplay', '-q', '/tmp/ocr_speech.wav'],
                check=True, timeout=10
            )
            # Clean up
            if os.path.exists('/tmp/ocr_speech.wav'):
                os.remove('/tmp/ocr_speech.wav')
            self.is_speaking = False
            return True
        except subprocess.TimeoutExpired:
            print("Speech timeout - skipping")
            self.is_speaking = False
            return False
        except Exception as e:
            print(f"Speech error: {e}")
            self.is_speaking = False
            return False
    
    def preprocess_image(self, roi):
        """Preprocess image for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding for better results in varying light
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        
        
        return thresh
    
    def clean_text(self, text):
        """Clean and filter OCR text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common OCR artifacts
        text = text.replace('|', 'I').replace('0', 'O')
        
        return text.strip()
    
    def draw_ui(self, frame, text, roi_coords):
        """Draw UI elements on frame"""
        x1, y1, x2, y2 = roi_coords
        
        # Draw ROI rectangle
        color = (0, 255, 0) if not self.is_speaking else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        
        status = "Speaking..." if self.is_speaking else "Ready"
        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display detected text
        if text:
            display_text = text[:50] + "..." if len(text) > 50 else text
            cv2.putText(frame, display_text, (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main application loop"""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*50)
        print("OCR Text-to-Speech Reader Active (Pi Cam via rpicam-vid)")
        print("="*50)
        print(f"Processing every {self.process_every} frames")
        print(f"Speak delay: {self.speak_delay} seconds")
        print(f"Minimum text length: {self.min_text_length} characters")
        print("Press 'q' to quit")
        print("="*50 + "\n")
        
        try:
            while True:
                frame = self.read_frame_from_rpicam()
                if frame is None:
                    print("Failed to grab frame from Pi camera")
                    break
                
                self.frame_count += 1
                
                
                height, width = frame.shape[:2]
                margin_w = int(width * (1 - self.roi_scale) / 2)
                margin_h = int(height * (1 - self.roi_scale) / 2)
                x1, y1 = margin_w, margin_h
                x2, y2 = width - margin_w, height - margin_h
                roi_coords = (x1, y1, x2, y2)
                
                text = ""
                
                
                if self.frame_count % self.process_every == 0 and not self.is_speaking:
                    # Extract ROI
                    roi = frame[y1:y2, x1:x2]
                    
                    # Preprocess image
                    processed = self.preprocess_image(roi)
                    
                    # Perform OCR
                    raw_text = pytesseract.image_to_string(
                        processed, 
                        config='--psm 6 --oem 3'
                    )
                    
                    text = self.clean_text(raw_text)
                    
                    # Auto-speak if conditions are met
                    current_time = time.time()
                    if text and text != self.last_spoken:
                        if current_time - self.last_speak_time > self.speak_delay:
                            print(f"\nDetected: {text}")
                            print("Speaking...")
                            
                            if self.speak(text):
                                self.last_spoken = text
                                self.last_speak_time = current_time
                                print("Done speaking\n")
                
                
                frame = self.draw_ui(frame, text if text else self.last_spoken, roi_coords)
                
                # Display frame
                cv2.imshow('OCR Text-to-Speech Reader', frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        cv2.destroyAllWindows()
        if os.path.exists('/tmp/ocr_speech.wav'):
            os.remove('/tmp/ocr_speech.wav')
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        print("Cleanup complete")

def main():
    reader = OCRReader()
    reader.run()

if __name__ == "__main__":
    main()
