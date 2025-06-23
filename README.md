This project is designed as an object detection system integrated into a mobile application using Vue.js, TensorFlow.js, and TensorFlow Lite. It utilizes the device's camera to capture images and performs real-time object detection using a pre-trained YOLOv8 model.

# Features of the Project:
## Real-time Object Detection:

- The application captures frames from the camera and processes them using the YOLOv8 model (converted to TensorFlow Lite for optimized performance on mobile devices).

- Detected objects are highlighted with bounding boxes, and their labels and confidence scores are displayed on the screen.

## Object Cropping and Labeling:

- After detecting objects, the application crops and stores images of each object, displaying them in a stack at the bottom of the screen.

- Each cropped object image is labeled, and the label is presented in Vietnamese based on a mapping file.

## Text-to-Speech Integration:

- The system reads out the labels of the detected objects aloud in Vietnamese using a text-to-speech (TTS) API, providing an accessible solution for users.

## TensorFlow Lite Model:

- The project leverages TensorFlow Lite for efficient object detection on mobile devices. The model is loaded and used to perform inference on the captured camera frames.

- The model processes the images in 640x640 resolution, utilizing letterboxing to maintain the aspect ratio of the images before passing them to the model.

Efficient Processing:

## The application runs an inference loop that captures frames from the camera, performs object detection, and displays the results at a smooth frame rate.

- The frames are preprocessed to fit the input size of the model and the results are drawn on a canvas overlaying the camera preview.

# Technical Stack:
- Frontend: Vue.js, Capacitor for mobile development

- AI: TensorFlow.js, TensorFlow Lite for model inference

- Text-to-Speech: Capacitor-community's Text-to-Speech plugin

- Model: YOLOv8 model converted to TensorFlow Lite format for real-time inference

# Application Flow:

1. Capture Frame: The app continuously captures frames from the camera.

2. Preprocessing: Each frame is preprocessed into a 640x640 image with proper padding to match the input dimensions required by the YOLOv8 model.

3. Inference: The preprocessed image is passed to the YOLOv8 model for object detection.

4. Post-processing: The results, including bounding box coordinates, class IDs, and confidence scores, are processed and drawn on the canvas overlay.

5. Object Cropping and Labeling: The detected objects are cropped and displayed in a stack with labels, along with the text-to-speech output.

This system provides an interactive and accessible object detection application, perfect for real-time scenarios such as security, inventory management, or user assistance.
