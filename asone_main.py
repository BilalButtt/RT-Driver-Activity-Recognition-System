import cv2
import asone
from asone import utils
from asone import ASOne

video_path = 'data/asad-behind.mp4'
output_video_path = 'data/asad-behind-detected.mp4'

# Initialize ASOne detector
detector = ASOne(detector=asone.YOLOV7_TINY_PYTORCH, use_cuda=False)  # Set use_cuda to False for CPU

filter_classes = ['person']  # Set to None to detect all classes

cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    dets, img_info = detector.detect(frame, filter_classes=filter_classes)

    bbox_xyxy = dets[:, :4]
    class_ids = dets[:, 5]

    # Draw bounding boxes on the frame
    frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=class_ids)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('result', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
