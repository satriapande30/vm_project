# Virtual Mouse Control Using Hand Poses 👋
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0F9D58?style=for-the-badge&logo=Google&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

Control your computer mouse using hand poses! This project uses computer vision and machine learning to transform your hand movements into mouse controls, creating a natural and intuitive interface for computer interaction.

## ✨ Features

- 🖱️ Control mouse cursor with hand movements
- 👆 Support for multiple hand poses:
| Gesture | Action |
|---------|--------|
| ✌️ V Pose | Mouse Movement |
| 👆 Middle Finger | Left Click |
| ☝️ Index Finger | Right Click |
| ✊ Fist | Start Drag |
| 🖐️ Palm | Release Drag |
- 🎯 Precise tracking within defined boundaries
- 🔄 Real-time gesture recognition
- 📊 Performance visualization and analytics
- 🎬 Smooth gesture transitions
- 📝 Comprehensive logging system

## 🛠️ Technology Stack

- **Python 3.x**: Core programming language
- **OpenCV**: Real-time image processing
- **MediaPipe**: Hand landmark detection
- **scikit-learn**: Machine learning for gesture recognition
- **PyAutoGUI**: Mouse control interface
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization

## 🎯 Project Structure

```
virtual-mouse-control/
├── data/
│   └── raw/               # Raw dataset images
│      ├──fist
│      ├──palm
│      ├──index_finger
│      ├──middle_finger
│      ├──v_pose
│      ├──v_pose_closed
│      └──random
├── output/
│   ├──error_images
│   ├── features/          # Extracted features
│   ├── log/              # System logs
│   ├── model/            # Trained models
│   └── visualization/    # Performance graphs
├──src/
│   ├── collect_dataset.py    # Dataset collection script
│   ├── processing_dataset.py # Dataset processing
│   ├── train_dataset.py      # Model training
│   └── virtual_mouse.py      # Main application
├── requirements.txt        # Dependencies
├── LICENSE                 # MIT License
├── README.MD
```

## 📋 Prerequisites

- Python 3.x
- Webcam
- Required Python packages:
  ```bash
  pip install opencv-python mediapipe numpy pandas scikit-learn pyautogui matplotlib seaborn
  ```

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/virtual-mouse-control.git
   cd virtual-mouse-control
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Collect training data**
   ```bash
   python collect_dataset.py
   ```

4. **Process the dataset**
   ```bash
   python processing_dataset.py
   ```

5. **Train the model**
   ```bash
   python train_dataset.py
   ```

6. **Run the virtual mouse system**
   ```bash
   python virtual_mouse.py
   ```

## 🎮 Usage Guide

1. **Tracking Area**: Keep your hand within the visible boundary box for optimal tracking
2. **Supported Gestures**:
   - Make a "V" sign to enable cursor tracking
   - Show middle finger for left click
   - Show index finger for right click
   - Make a fist to start dragging
   - Show palm to release drag

## 📊 Performance Metrics

The system provides comprehensive performance visualization including:
- Confusion matrix
- Classification report
- Model performance comparison
- Real-time FPS counter

### Architecture:
1. **Hand Detection**: Uses MediaPipe for real-time hand landmark detection
2. **Feature Extraction**: Calculates distances between key landmarks
3. **Gesture Classification**: SVM model classifies hand poses
4. **Motion Tracking**: Smooth tracking with boundary detection
5. **Action Execution**: Converts gestures to mouse actions


## 📝 Logging System

The application maintains detailed logs for:
- System initialization
- Gesture recognition events
- Mouse actions
- Error tracking
- Performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe team for their excellent hand tracking solution
- OpenCV community for comprehensive computer vision tools
- scikit-learn team for machine learning capabilities

## 📧 Contact

Satria Pande - [@satriapande30](https://www.instagram.com/satriapande30/) - satria.pande30@gmail.com

Project Link: [https://github.com/satriapande30/vm_project/](https://github.com/satriapande30/vm_project/)
