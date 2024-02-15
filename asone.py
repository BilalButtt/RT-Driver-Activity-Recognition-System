import asone
from asone import ASOne

# Instantiate Asone object
detect = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOV7_PYTORCH, use_cuda=True) #set use_cuda=False to use cpu

filter_classes = ['person'] # set to None to track all classes

# ##############################################
#           To track using video file
# ##############################################
# Get tracking function
track = detect.track_video('data/sample_videos/test.mp4', output_dir='data/results', save_result=True, display=True, filter_classes=filter_classes)

# Loop over track to retrieve outputs of each frame
for bbox_details, frame_details in track:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    # Do anything with bboxes here

# ##############################################
#           To track using webcam
# ##############################################
# Get tracking function
track = detect.track_webcam(cam_id=0, output_dir='data/results', save_result=True, display=True, filter_classes=filter_classes)

# Loop over track to retrieve outputs of each frame
for bbox_details, frame_details in track:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    # Do anything with bboxes here

# ##############################################
#           To track using web stream
# ##############################################
# Get tracking function
stream_url = 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4'
track = detect.track_stream(stream_url, output_dir='data/results', save_result=True, display=True, filter_classes=filter_classes)

# Loop over track to retrieve outputs of each frame
for bbox_details, frame_details in track:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    # Do anything with bboxes here