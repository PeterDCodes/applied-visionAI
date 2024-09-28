
#Program to generate a heatmap result of a specific class from a model

#demo model and video
video_path = 'video_3.mp4'
model_path = './Model1/train/weights/last.pt'

#Input array here the specific class that will be conisdered in heatmap
classes_for_heatmap = [0]  # array of classes for heatmap [0,1,2,etc]


import cv2

from ultralytics import YOLO, solutions

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("heatmap_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init heatmap
heatmap_obj = solutions.Heatmap(
    colormap=cv2.COLORMAP_PARULA,
    view_img=True,
    shape="circle",
    names=model.names,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False, classes=classes_for_heatmap)

    im0 = heatmap_obj.generate_heatmap(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()