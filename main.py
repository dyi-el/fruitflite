import cv2
import av
import numpy as np
import streamlit as st

import tensorflow as tf

from utils.iceServer import get_ice_servers
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer


logo = Image.open('assets/fruiTFLite.png')
logo = logo.resize((200, 200))
st.image(logo)


# Downloaded TFLite model and labels paths
MODEL_PATH = "assets/model.tflite"

# Load labels from file
CLASSES = {
    0: "dorsalis",
    1: "occipitalis"
}

# Load TFLite model
interpreter = tf.lite.Interpreter(MODEL_PATH)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]

# Predefined label colors
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Score threshold slider
score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)

# Callback function for processing video frames
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    

    # Prepare the input data
    input_data = cv2.flip(image, 1)
    input_data = cv2.resize(input_data, (width, height))
    input_data = np.expand_dims(input_data, axis=0)

    # Set frame as input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Perform inference
    interpreter.invoke()

    # Get output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Convert the output array into a structured form.
    detections = [
        {
            "class_id": int(classes[i]),
            "label": CLASSES[int(classes[i])],
            "score": float(scores[i]),
            "box": (boxes[i] * np.array([width, height, width, height])).astype(int),
        }
        for i in range(len(scores))
        if (scores[i] > score_threshold) and (scores[i] <= 1.0)
    ]

    # Render bounding boxes and captions
    for detection in detections:
        caption = f"{detection['label']}: {round(detection['score'] * 100, 2)}%"
        color = COLORS[detection['class_id']]

        # Adjust bounding box coordinates for rotated image
        xmin, ymin, xmax, ymax = detection['box']
        xmin, ymin, xmax, ymax = height - ymax, xmin, height - ymin, xmax

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            image,
            caption,
            (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return av.VideoFrame.from_ndarray(image, format="bgr24")

# Define the WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {
                                    "width": 320,
                                    "height": 320
                                }, 
                              "audio": False},
    video_html_attrs={
                "style" : {"width":"100%"},
                "controls": False,
                "muted" : True,
                "autoPlay": True,
                },
    async_processing=True,
)
