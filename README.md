# Autonomous Drone for Object Detection and Tracking Using YOLOv8

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![Hardware](https://img.shields.io/badge/Drone-DJI%20Ryze%20Tello-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

This repository contains the source code for my Bachelor's Final Project (Tugas Akhir) at **Institut Teknologi Sepuluh Nopember (ITS)**, titled *"Autonomous Drone for Object Detection and Tracking Using YOLOv8"*.

The system enables a **DJI Ryze Tello** drone to autonomously detect, track, and follow a specific target (a toy car) in real-time. By utilizing **YOLOv8** for high-speed object detection and a custom control logic loop, the drone adjusts its yaw, altitude, and distance to maintain the target within the frame and follow it from behind.

## 🚀 Key Features

* **Real-Time Object Detection**: Utilizes **YOLOv8n (Nano)** optimized for speed to detect objects with low latency.
* **Intelligent Tracking Logic**: The drone calculates the center of the bounding box relative to the frame center and adjusts flight parameters (yaw, throttle, pitch) to center the target.
* **Orientation Recognition**: Trained on a custom dataset to recognize 3 specific orientations of the target car to determine the correct pursuit angle:
    * `Car Toy Behind` (Target)
    * `Car Toy Left`
    * `Car Toy Right`].
* **Multi-Threading Architecture**: Implements 4 parallel threads for efficiency:
    1.  Video Streaming
    2.  YOLOv8 Detection
    3.  Drone Movement Command
    4.  Manual Override/Keyboard Control

## 🛠️ Tech Stack & Hardware

### Hardware
* **Drone**: DJI Ryze Tello
* **Processing Unit**: Laptop (Tested on Legion 5, GTX 1660 Ti for training/inference).

### Software & Libraries
* **Language**: Python
* **Computer Vision**: OpenCV, Ultralytics YOLOv8
* **Drone Control**: `djitellopy` SDK.
* [**Dataset Management**: Roboflow (for annotation and augmentation).

## ⚙️ How It Works

1.  **Acquisition**: The drone streams video (720p/480p) to the host computer via Wi-Fi.
2.  **Detection**: YOLOv8 inference runs on each frame to identify the target and its orientation class.
3.  **Control Loop**:
    * If the target is `Car Toy Left`, the drone moves right and rotates to get behind it.
    * If the target is `Car Toy Right`, the drone moves left and rotates.
    * If the target is `Car Toy Behind`, the drone maintains distance using visual feedback (moving forward/backward).
4.  **Actuation**: Commands are sent back to the drone via UDP using the `djitellopy` library.

## 📊 Performance Results

Based on the final testing results presented in the thesis:

* **Model Accuracy (mAP50)**: 0.993 after 150 epochs of training.
* **Detection Range**:
    * **2-3 Meters**: 100% detection accuracy.
    * **4 Meters**: ~90% accuracy with reliable tracking.
    * **5+ Meters**: Detection drop-off observed.
* **Control Response**: The system successfully maintains a stable hover and tracking mechanism using feedback control loops.

## 💻 Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Ezekielna70/TADJITello.git](https://github.com/Ezekielna70/TADJITello.git)
    cd TADJITello
    ```

2.  **Install Dependencies**
    Ensure you have Python installed. It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Key requirements include `ultralytics`, `djitellopy`, `opencv-python`, and `numpy`.*

3.  **Connect to Drone**
    * Turn on the DJI Tello.
    * Connect your computer's Wi-Fi to the Tello's network.

4.  **Run the System**
    ```bash
    python main.py
    ```
    *(Note: Replace `main.py` with your actual main script name if different)*

5.  **Controls**
    * ]**Q**: Land and Quit.
    * **T**: Takeoff (if manual trigger is enabled).
    * The system typically performs auto-takeoff after stream initialization.

## 📷 Gallery / Demo

![Picture1](https://github.com/user-attachments/assets/ae371786-6297-4947-804a-8208acb25282)
<img width="1234" height="992" alt="image" src="https://github.com/user-attachments/assets/3b112109-a880-48b6-834e-f825e029d064" />
<img width="1249" height="1005" alt="image" src="https://github.com/user-attachments/assets/82e22cce-bff4-44bb-90f5-f80fba072e9c" />
<img width="1252" height="987" alt="image" src="https://github.com/user-attachments/assets/5e536f30-c7c5-4add-a912-d0c235abf0ba" />



## 👨‍💻 Author

**Ezekiel Walfred Ebenezer Pangihutan Napitupulu**
* **Institution**: Institut Teknologi Sepuluh Nopember (Computer Engineering)
* **LinkedIn**: [linkedin.com/in/ezekiel-na70](https://www.linkedin.com/in/ezekiel-na70/)
* **Email**: ewalfredna70@gmail.com

---
*This project was submitted as a requirement for the Bachelor's Degree in Computer Engineering at ITS Surabaya, 2025.*
