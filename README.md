**Spectrum Care: Autism Spectrum Disorder (ASD) Predictor**

PredictorSpectrum Care is a multi-modal, deep-learning-based screening system designed for the early detection of Autism Spectrum Disorder (ASD). It integrates facial structure analysis, real-time eye gaze tracking, and behavioral questionnaires into a unified web-based platform to provide a scalable and accessible diagnostic tool.

**Project Link**
https://huggingface.co/spaces/NoorEJannat/spectrum-care

**Project Overview**

Early diagnosis of ASD is critical for effective intervention, yet traditional clinical methods are often time-consuming and expensive. Spectrum Care addresses these challenges by offering a real-time, low-cost screening tool that can be used by caregivers and healthcare professionals across various devices.

**Key Features**

Multi-Modal Detection: Combines three distinct diagnostic layers—facial image analysis, eye gaze tracking, and behavioral assessment—to improve prediction accuracy and reliability.
Deep Learning Models: Utilizes state-of-the-art architectures for image-based classification, including ViT-B/16 (highest performing with 91.67% accuracy), ConvNeXt-Tiny, EfficientNet-B4, ResNet-50, and VGG19.
Real-Time Eye Gaze Tracking: Integrated WebGazer.js for immediate analysis of visual attention patterns during user interaction.
Behavioral Assessment: Includes a structured questionnaire to evaluate clinical traits associated with ASD.
Assistive Chatbot: Features a Dialogflow-powered chatbot to guide users through the screening process.

**Technical Architecture**

Frontend: Developed using modern web technologies, incorporating WebGazer.js for client-side gaze tracking and a user-friendly interface for image uploads and surveys.
Backend: Powered by Flask (Python) and PyTorch for managing model inferences and processing user data.
Dataset: Trained on a custom-curated dataset from Kaggle of 2,936 images (autistic and non-autistic classes) with extensive data augmentation (rotation, flipping, normalization) for robustness.

**Performance Metrics (ViT-B/16 Model)**

Accuracy: 91.67% 
AUC: 0.970 
Optimization: Trained using the AdamW optimizer with an adaptive learning rate schedule to prevent overfitting.

**Installation & Environment**
The project was developed using the following environment:

Frameworks: PyTorch, TensorFlow 
Backend: Flask 
Compute: Trained using Google Colab's T4 GPUs and Kaggle’s GPU facilities 

**Ethical Considerations**
Privacy and ethical integrity were prioritized. All data used for training and inference were anonymized, and image samples were used strictly for academic purposes.

**Future Work**
Incorporating additional data modalities such as speech and environmental audio.
Developing more advanced assistive technologies for immediate post-diagnosis intervention.

Author: Noor E Jannat Neha 
Supervisor: Dr. Fizar Ahmed, Daffodil International University 
