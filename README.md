# 🚀 AirControl — Vision-Based Gesture Control System

**Control your Windows PC using natural hand gestures.**

AirControl is a Python-based, real-time computer vision application that transforms hand movements into desktop interactions. It combines webcam-based hand tracking, gesture recognition and Windows automation to explore a more natural way of interacting with computers.

---

## 🎥 Project Demo

<!-- Add your demo video link or a GIF here -->

**Demo video:** Coming soon.

---

## ✨ Features

### 👁️ Real-Time Hand Tracking
- Webcam-based hand detection.
- 21-point hand landmark tracking.
- Support for tracking up to two hands.
- Live visualisation of hand landmarks and connections.

### 🖱️ Air Mouse
- Control the mouse cursor using hand movements.
- Gesture-based clicking.
- Pinch-and-hold drag interactions.
- Gesture-based scrolling.
- Cursor smoothing and sensitivity controls.

### 🧠 Gesture Intelligence
- Static hand-pose recognition.
- Dynamic gesture recognition.
- Open-palm and fist detection.
- Finger-based gesture classification.
- Gesture-based interaction logic.

### 🖥️ Windows Desktop Automation
- Gesture-triggered desktop actions.
- Window management.
- System-level controls.
- Configurable gesture-to-action mappings.

### 🧊 Experimental 3D Hand Tracking
- 3D hand landmark processing.
- Articulated skeletal hand representation.
- Spatial gesture recognition development.
- Experimental 3D hand visualisation.

*Note: The 3D tracking subsystem is experimental and is being refined. Its depth estimates are derived from a monocular webcam rather than a dedicated depth sensor.*

### 🎨 Futuristic Dashboard
- Dark-themed control interface.
- Live camera viewport.
- Hand tracking telemetry.
- FPS monitoring.
- Air Mouse and gesture controls.
- Tracking and automation status indicators.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| OpenCV | Webcam capture and image processing |
| MediaPipe | Hand detection and landmark estimation |
| NumPy | Numerical computations |
| CustomTkinter | Desktop dashboard |
| Pillow | Image processing and display |
| PyAutoGUI | Mouse and keyboard automation |

---

## 🏗️ System Architecture

AirControl is organised into modular components:

```text
AirControl/
│
├── main.py
├── config.py
├── requirements.txt
│
├── vision/
│   ├── camera.py
│   └── hand_tracker.py
│
├── interface/
│   └── dashboard.py
│
├── input_control/
│   ├── air_mouse.py
│   ├── gesture_controller.py
│   └── cursor_smoother.py
│
├── hand3d/
│   ├── hand_model.py
│   ├── skeleton.py
│   ├── joint_angles.py
│   ├── hand_orientation.py
│   └── spatial_features.py
│
├── utils/
│   └── fps.py
│
└── assets/
    └── hand_landmarker.task
```

*The structure above illustrates the intended modular architecture. Refer to the actual repository files for the current implementation.*

### Processing Pipeline

1. Webcam captures live video.
2. OpenCV processes camera frames.
3. MediaPipe estimates hand landmarks.
4. The gesture engine interprets hand poses and movements.
5. The input controller maps recognised gestures to interactions.
6. The automation engine executes supported desktop commands.
7. The dashboard displays live tracking and system telemetry.

---

## ⚙️ Installation

### Requirements

- Windows 10 or Windows 11
- Python 3.11 or later
- Webcam or compatible virtual webcam
- Working internet connection for initial dependency installation

### 1. Clone the repository

```bash
git clone https://github.com/udaygb890-commits/AirControl.git
```

### 2. Navigate into the project

```bash
cd AirControl
cd AirControl
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Launch AirControl

```bash
python main.py
```

Make sure the required MediaPipe hand landmark model is present in the expected assets directory.

---

## 🧪 Current Development Status

| Component | Status |
|---|---|
| Camera and vision engine | Implemented |
| Hand landmark tracking | Implemented |
| Air Mouse | Implemented |
| Gesture recognition | Implemented |
| Windows desktop automation | Implemented |
| 3D hand representation | Experimental |
| Spatial gesture improvements | In development |

---

## 🔮 Future Improvements

- Improved gesture recognition accuracy.
- More robust 3D hand-pose estimation.
- Better spatial interaction.
- Customisable gesture profiles.
- Improved tracking under different lighting conditions.
- Further performance optimisation.

---

## 👨‍💻 Developer

**Uday G B**

GitHub: [@udaygb890-commits](https://github.com/udaygb890-commits)

Project Repository: [AirControl](https://github.com/udaygb890-commits/AirControl)

---

## 📜 License

A license has not yet been selected for this project. Please contact the author before reusing or redistributing the code.

---

**Built with Python, Computer Vision and a vision for the future of human-computer interaction.** 🚀
