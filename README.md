# Brain Tumor Detection Using Deep Learning  

A web-based application that detects brain tumors in MRI images using deep learning techniques. This project was initially developed as a semester project for the Deep Learning course and is now being enhanced as a Final Year Project.  

---

## Features  
- Detects the presence of brain tumors in MRI images.  
- Interactive web-based interface for easy usage.  
- Upload MRI scans and receive instant predictions.  
- Aimed to assist radiologists and doctors in making accurate diagnoses.  

---

## Dataset  
- **Number of Images**: 5000 MRI scans (tumor and non-tumor).  
- Preprocessed and augmented to improve model performance.  

---

## Technology Stack  
### Frontend  
- HTML, CSS, JavaScript (for user interface).  
### Backend  
- Python, Flask (for model integration and API development).  
### Model  
- Convolutional Neural Network (CNN) architecture.  
- Libraries: TensorFlow, Keras, NumPy, Pandas, OpenCV.  

---

## Training Details  
- **Training Dataset**: 80% of the images used for training.  
- **Validation Dataset**: 20% of the images used for validation.  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score.  

---

## Installation  
1. Clone this repository:  
   ```bash  
   git clone https://github.com/your-username/brain-tumor-detection.git  
cd brain-tumor-detection  

pip install -r requirements.txt  

python app.py  
