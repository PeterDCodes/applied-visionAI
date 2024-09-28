
#this program will count the number of detected object in a video

video_path = 'video_2.mp4'
model_path = 'model.pt'

import cv2

from ultralytics import YOLO, solutions

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

#Position line in bottom center for demo video to simuilate incoming object to station
region_points = [(int(w/2), int(h/2)), (int(w/2),h),(int(w/2)+100,h), (int(w/2)+100, int(h/2))]  # line or region points. line would be 2 coordinate pairs, region would be 4 to make a box
classes_to_count = [0]  # clock class

# Video writer
video_writer = cv2.VideoWriter("results/object_counter.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=region_points,
    names=model.names,
    draw_tracks=False,
    line_thickness=2,
    
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)
    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()