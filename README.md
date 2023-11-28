<img src="/assets/fruiTFLite.png" alt="fruitflite" width="300"/>

# fruiTFLite: Object Detection for Bactocera Dorsalis and Bactocera Occipitalis


## Overview

This project demonstrates real-time object detection for Bactocera Dorsalis and Bactocera Occipitalis using a TensorFlow Lite model. The implementation is done using Streamlit for the user interface and TensorFlow Lite for efficient and fast inference. The model is trained using AutoML. Feel free to use the model.

## Requirements

The exact requirements is described in the Pipfile

- Python 3.x
- Streamlit
- TensorFlow
- OpenCV
- AV (Audio/Video library)
- Numpy

##Usage

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

## License

This project is licensed under the MIT License.