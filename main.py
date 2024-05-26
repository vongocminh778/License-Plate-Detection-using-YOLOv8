import cv2 # Import OpenCV library for image processing
import torch  # Import PyTorch library for deep learning
import threading # Import threading for multithreading support
import time # Import time for sleep function
import serial  # Import serial library for serial communication
from cnocr import CnOcr  # Import CnOcr for text recognition
from core import deskew # Import deskew function for deskewing images
import re # Import re for regular expression operations
from ultralytics import YOLO # Import YOLO for object detection
from ultralytics.utils.plotting import Annotator, colors  # Import Annotator and colors for plotting

class ObjectDetection:
    def __init__(self):
        self.data_lock = threading.Lock()
        self.terminate_flag = False
        self.ocr = CnOcr()  # init ocr
        self.text_output = None

    def process_ocr_output(self, text):
        text = text[0]['text']
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
        if len(text) == 8 and text[:2].isdigit() and text[2].isalpha() and text[-5:].isdigit():
            return text

    def start_detection(self, run_event):
        # used to record the time when we processed last frame 
        prev_frame_time = 0
        
        # used to record the time at which we processed current frame 
        new_frame_time = 0
        # Initialize YOLOv8 model
        model = YOLO("./models/best.pt")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Initialize the camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Set frame dimensions
        cap.set(3, 640)  # Width
        cap.set(4, 480)  # Height

        # while not self.terminate_flag:
        while run_event.is_set():
            new_frame_time = time.time() 
            # Read a frame from the camera
            ret, img = cap.read()
            if not ret:
                break

            results = model(img, conf=0.5, stream=True, device=device, verbose=False)

            for r in results:
                boxes = r.boxes.xyxy.cpu()
                clss = r.boxes.cls.cpu().tolist()

                if len(boxes) > 0:
                    xmin, ymin, xmax, ymax = boxes.numpy().astype(int)[0]
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img.shape[1], xmax)
                    ymax = min(img.shape[0], ymax)
                    cropped_image = img[ymin:ymax, xmin:xmax]
                    corrected_img = deskew(cropped_image)
                    text_output = self.ocr.ocr(corrected_img)
                    if text_output:
                        text_output = self.process_ocr_output(text_output)
                        with self.data_lock:
                            if text_output and text_output != "None": self.text_output = text_output
                        print(text_output)

                confidences = r.boxes.conf.cpu().tolist()
                annotator = Annotator(img, line_width=2, example=str(model.model.names))

                for box, cls, conf in zip(boxes, clss, confidences):
                    annotator.box_label(box, str(model.model.names[cls]) + f" {conf:.2f}", color=colors(cls, True))

            fps = 1/(new_frame_time-prev_frame_time) 
            prev_frame_time = new_frame_time 
        
            # converting the fps into integer 
            fps = int(fps) 
        
            # converting the fps to string so that we can display it on frame 
            # by using putText function 
            fps = str(fps) 
        
            # putting the FPS count on the frame 
            cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA) 
            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == ord('q'):
                self.terminate_flag = True

        cap.release()
        cv2.destroyAllWindows()

    def send_serial(self, run_event):
        try:
            lst_plate = ['51F04877.', '51D10039.', '51F02687.', '51G10096.', '51A05227.', '51F59011.', '51A85325.', '51A60216.']
            lst_alphabet = ['1', '2', '3', '4', '5', '6', '7', '8']
            Ser = serial.Serial('COM4', 115200)
            time.sleep(2)
            connected = Ser.is_open
            if connected:
                print("Serial connection is open.")
            else:
                print("Serial connection is not open. Check your serial port.")

            while run_event.is_set():
                with self.data_lock:
                    data = str(self.text_output) + "."
                    self.text_output = None
                if connected and data != "None.":
                    # Find the index in lst_plate
                    if data in lst_plate:
                        index = lst_plate.index(data)
                        data_write = lst_alphabet[index]
                        Ser.write(data_write.encode())
                        print(f"The index of data in lst_plate is: {data} - {data_write}")
                    else:
                        print("Data not found in lst_plate.")

                    time.sleep(0.5)
                    print("data: {}".format(data))
                time.sleep(0.5)
        except serial.serialutil.SerialException as e:
                print("Error:", e)
                print("Serial port is not available or cannot be opened. Please check your serial port configuration.")

def main():
    object_detection = ObjectDetection()

    run_event = threading.Event()
    run_event.set()

    t1 = threading.Thread(target=object_detection.start_detection, args=(run_event,))
    t2 = threading.Thread(target=object_detection.send_serial, args=(run_event,))

    t1.start()
    t2.start()

    try:
        while not object_detection.terminate_flag:
            time.sleep(.1)
    except KeyboardInterrupt:
        pass
    finally:
        run_event.clear()
        t1.join()
        t2.join()
    print("Threads successfully closed")

if __name__ == "__main__":
    main()
