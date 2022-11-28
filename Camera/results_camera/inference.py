import numpy as np
import cv2
import pyrealsense2 as rs
import torch
import time

# from Camera.results_camera.models.create_fasterrcnn_model import create_model
# from Camera.results_camera.utils.annotations import inference_annotations
# from Camera.results_camera.utils.transforms import infer_transforms  

from Camera.results_camera.models.create_fasterrcnn_model import create_model
from Camera.results_camera.utils.annotations import inference_annotations
from Camera.results_camera.utils.transforms import infer_transforms  

def camera_inference():
    
    # Weights model:
    best_model_path="Camera/results_camera/ball_can_bottle_tom_32_640x640.pth"

    # Detection threshold
    detection_threshold = 0.7
    data_configs = None

    # Select device
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load weights if path provided.   
    checkpoint = torch.load(best_model_path, map_location=DEVICE)

    # If config file is not given, load from model dictionary.
    if data_configs is None:
        data_configs = True
        NUM_CLASSES = checkpoint['config']['NC']
        CLASSES = checkpoint['config']['CLASSES']

    build_model = create_model[checkpoint['model_name']]
    model = build_model(num_classes=NUM_CLASSES, coco_model=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
   
    # Camera number id
    device_number = '045322072493'
    # Configure camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device_number) # This number is the serial number of the camera connected,

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    # pipeline_profile = config.resolve(pipeline_wrapper)
    

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    ## Device product line is device_product_line == 'D400'

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    offset = [[350,400],[500,600]]
    # offset = [[0,480],[0,640]]
    i=0
    while i<100:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
  
        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', color_image)
        # cv2.waitKey(1)
        i+=1


    # while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    image = np.asanyarray(color_frame.get_data())

    # image = cv2.imread(image)
    orig_image = image.copy()
    crop_im = orig_image[offset[0][0]:offset[0][1],offset[1][0]:offset[1][1]]
    
    # BGR to RGB
    image = cv2.cvtColor(crop_im, cv2.COLOR_BGR2RGB)
    image = infer_transforms(image)


    # Add batch dimension.
    image = torch.unsqueeze(image, 0)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image.to(DEVICE))
    end_time = time.time()

    # Load all detection to CPU for further operations.
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # Carry further only if there are detected boxes.
    if len(outputs[0]['boxes']) != 0:
        orig_image = inference_annotations(
            outputs, detection_threshold, CLASSES,
            COLORS, orig_image, offset
        )
        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(2000)
        
    print("Time taken for prediction: " + str(end_time-start_time))
    print('TEST PREDICTIONS COMPLETE')

    boxes = outputs[0]['boxes'].data.numpy()
    labels = outputs[0]['labels'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    labels = labels[scores >= detection_threshold].astype(np.int32)
    scores = scores[scores >= detection_threshold].astype(np.int32)
    pred_classes = [CLASSES[i] for i in labels]
    return pred_classes, orig_image
        
        
# if __name__ == '__main__':
#     # args = parse_opt()
#     main()

# pred_classes, orig_image  = camera_inference()