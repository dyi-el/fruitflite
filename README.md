<img src="/assets/fruiTFLite.png" alt="fruitflite" width="300"/>

# fruiTFLite: Object Detection for Bactocera Dorsalis and Bactocera Occipitalis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fruitflite.streamlit.app)
## Overview

This project demonstrates real-time object detection for fruitflies (Bactocera Dorsalis and Bactocera Occipitalis) using a TensorFlow Lite model. The implementation is done using Streamlit for the user interface and TensorFlow Lite for efficient and fast inference. The model is trained using AutoML. Feel free to use the model.

## Requirements

The exact requirements is described in the Pipfile

- Python 3.x
- Streamlit
- TensorFlow
- OpenCV
- AV (Audio/Video library)
- Numpy

## Usage

To install the dependencies with pipenv:
*Assuming you have pipenv already installed*

```bash
pipenv install
```

Run the Streamlit app with the following command:

```bash
streamlit run main.py
```

Open the provided link in your web browser to access the real-time object detection application.

Adjust the score threshold using the slider to control the detection sensitivity.

#### YOU CAN CHANGE THE MODEL TO INFERENCE YOUR OWN TFLITE MODEL

## Android App
The android app is has bugs in terms of processing images. It is a modified code from ["Build and deploy a custom object detection model with TensorFlow Lite (Android)"](https://developers.google.com/codelabs/tflite-object-detection-android). 

![Sample-Screen](sample-screen.gif)

## Acknowledgement
Much of this project is derived from other open-source projects such as
- [TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/LICENSE) by EdjeElectronics
- [streamlit-webrtc/pages/1_object_detection.py](https://github.com/whitphx/streamlit-webrtc/blob/main/pages/1_object_detection.py) by whitphx
- [Build and deploy a custom object detection model with TensorFlow Lite (Android)](https://developers.google.com/codelabs/tflite-object-detection-android) by Google

## License

This project is licensed under the MIT License.
