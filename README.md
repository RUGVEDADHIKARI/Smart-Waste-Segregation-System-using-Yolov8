# 🚮 Smart Waste Segregation System using YOLOv8

Welcome to the Smart Waste Segregation System! This project leverages the power of YOLOv8, a state-of-the-art object detection model, to automate and optimize the process of waste classification and segregation.

---

## 🌟 Project Overview

The Smart Waste Segregation System is designed to:
- **Detect and classify waste** Detecting the waste and classifying into recyclable, hazardious and desposable in real time using computer vision.
- **Automate sorting** for improved recycling efficiency and reduced human intervention.
- **Enable scalability** for smart cities, waste management facilities, and eco-friendly initiatives.

---

## 🛠️ Tech Stack

- **Python** (core logic and scripting)
- **YOLOv8** (object detection and classification)
- **OpenCV** (image processing)
- **Streamlit** (User Interface)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/RUGVEDADHIKARI/Smart-Waste-Segregation-System-using-Yolov8.git
cd Smart-Waste-Segregation-System-using-Yolov8
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Download YOLOv8 Weights

- Download the pre-trained YOLOv8 weights or train your own model.
- Place the weights file in the appropriate directory (specify path if needed).

### 4. Run the System

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
Smart-Waste-Segregation-System-using-Yolov8/
├── data/              # Dataset and annotations
├── best.pt            # YOLOv8 models/weights
├── app.py             # main file
├── helper.py          # processing file
├── settings.py        # file settings
├── outputs/           # Output images and videos
├── requirements.txt   # List of dependencies
└── README.md
```

---

## 🧪 Example Usage

- Upload or stream an image/video of waste.
- The system detects and classifies each item with confidence score.
  
## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or suggestions.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

[MIT License](LICENSE)

---

## 🙋‍♂️ Contact

- **GitHub:** [RUGVEDADHIKARI](https://github.com/RUGVEDADHIKARI)
- **LinkedIn:** [www.linkedin.com/in/rugved-adhikari]

---

> Empowering smart cities with intelligent waste management.  
> Built with ❤️ using YOLOv8 and Python.
