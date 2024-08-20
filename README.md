# Defect Detection with Tensorflow Lite and OpenCV

The ML source code and dataset mentioned in the paper "Integrating Machine Learning Model and Digital Twin System for Additive Manufacturing"
EfficientDet-Lite for 
1. Install and setup
2. Collect Images and Label
3. Training models
4. Detecting Defects
5. Freezing and Conversion
6. Performance Tuning
7. Training on Colab
8. Projects

![image](https://github.com/user-attachments/assets/81964c3a-2be5-43f1-8b52-27e9f0a41443)

![image](https://github.com/user-attachments/assets/7fb062e4-a261-4f78-946d-f5727c192a8f)

# Real-Time Defect Detection Using EfficientDet-Lite with Tensorflow Lite and OpenCV

## Introduction
This project focuses on real-time defect detection using various deep learning models, including EfficientDet-Lite, ResNet, and VGG16. The primary objective is to compare the performance of these models in detecting defects in real-time using different hardware setups, such as Raspberry Pi and Windows environments.

## Introduction

This repository contains the machine learning component of a research project focused on enhancing additive manufacturing through the integration of digital twin technology. Additive manufacturing, also known as 3D printing, is a promising manufacturing process with a wide range of applications. However, ensuring the quality and reliability of the manufactured products remains a significant challenge.

To address these challenges, this research proposes a digital twin system framework that integrates machine learning models for real-time defect detection and monitoring. The system employs a combination of Unity, OctoPrint, and Raspberry Pi to facilitate real-time control and monitoring of the additive manufacturing process. The machine learning models developed in this project have demonstrated high efficiency, achieving an Average Precision (AP) score of 92%, with performance metrics of 91% for defected objects and 94% for non-defected objects.

![image](https://github.com/user-attachments/assets/0d76660e-da3a-4d2e-aafb-0ae5cfcc64c4)

This repository provides the implementation of the machine learning models used in the digital twin system. It includes the code, datasets, and results that contributed to the defect detection component of the proposed framework. The overall goal is to enhance the quality and reliability of additive manufacturing by leveraging the capabilities of digital twins and machine learning.

![image](https://github.com/user-attachments/assets/715c0f85-b076-488b-8ab7-bc83cdaa3946)

In this repository, you will find:
- The source code for the machine learning models, including EfficientDet-Lite, ResNet, and VGG16, which were employed for defect detection.
- Scripts for processing videos and extracting frames for model testing.
- Detailed instructions on how to set up and run the machine learning components of the digital twin system.
- Results and performance metrics demonstrating the effectiveness of the models in real-time defect detection.

This work is part of a larger research effort aimed at advancing the field of digital twin technology in additive manufacturing, and we hope it provides a valuable resource for researchers and practitioners working in this area.

## Objectives
- Achieve accurate real-time defect detection on resource-constrained devices.
- Compare the performance of EfficientDet-Lite with other models like ResNet and VGG16.
- Evaluate the effectiveness of transfer learning techniques in defect detection tasks.
- Provide a comprehensive implementation that can be easily reproduced and extended.

## Dataset
The dataset used in this project consists of images and videos capturing various defect scenarios. Preprocessing steps included resizing, normalization, and data augmentation to enhance the model's ability to generalize across different defect types.

## Models
Several models were employed with transfer learning method in this project:
- **EfficientDet-Lite**: Chosen for its balance between accuracy and computational efficiency, making it ideal for real-time applications on devices with limited resources.
- **ResNet**: Used for comparison due to its established performance in image classification tasks.
- **VGG16**: Another comparative model known for its simplicity and effectiveness in transfer learning scenarios.

The chosen pre-trained model was fine-tuned using the collected dataset of 120 images related to additive manufacturing. The last few layers of the chosen model were replaced with new layers to accommodate the specific number of object classes relevant to additive manufacturing. During fine-tuning, the weights of the chosen pre-trained model were frozen, and only the newly added layers were trained using the collected dataset.

## Implementation Details
### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Real-Time-Defect-Detection-EfficientDet-Lite.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Real-Time-Defect-Detection-EfficientDet-Lite
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Running the Code:
   ```bash
   python train.py --model efficientdet-lite --dataset /path/to/dataset
   ```

## Video Processing
### Purpose
Videos were used to test the real-time capabilities of the models. Frames were extracted from videos to evaluate the models' performance on sequential data.

### Video Cutting Script
The following script was used to extract frames from video files:
```bash
import cv2

capture = cv2.VideoCapture('C:/Users/Admin/Documents/RA_3D_printer_project/trials/Cube 3D Printing 1.mp4')
frameNr = 0

while True:
    success, frame = capture.read()
    if success:
        cv2.imwrite(f'C:/Users/Admin/Documents/RA_3D_printer_project/trials/output/frame_{frameNr}.jpg', frame)
    else:
        break
    frameNr += 1

capture.release()

```
### Usage Instructions
1. Modify the video file path in the script to point to your video.
2. Run the script to extract frames, which will be saved in the specified output directory.

   
## Results
The following are the key performance metrics for each model:

- **EfficientDet-Lite**: Achieved a balance between accuracy and speed, making it ideal for real-time detection on Raspberry Pi.
- **ResNet**: Provided high accuracy but at the cost of slower processing speeds.
- **VGG16**: Showed moderate performance, with slower speeds and slightly lower accuracy compared to EfficientDet-Lite.

Comparison tables and charts summarizing the results are available in the `/results` directory.

## Future Work
- Explore additional lightweight models for even faster real-time detection.
- Optimize the current implementation for better performance on Raspberry Pi.
- Extend the dataset to include more diverse defect types and scenarios.

## References

Papers, libraries, and datasets used in this project include:

- [EfficientDet](https://arxiv.org/abs/1911.09070)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [VGG16](https://arxiv.org/abs/1409.1556)
- [OpenCV](https://opencv.org/)


