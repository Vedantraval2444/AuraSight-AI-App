# 👁️ AuraSight AI: Explainable AI for Diabetic Retinopathy Detection

A full-stack web application that provides instant, AI-powered screening for Diabetic Retinopathy (DR) using retinal fundus images. This tool not only classifies the severity of DR but also provides visual explanations of the AI's decision-making process through Grad-CAM heatmaps.

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Journey & Challenges Faced](#-project-journey--challenges-faced)
- [Getting Started](#-getting-started)
- [Disclaimer](#-disclaimer)

---

## ❗ Problem Statement

Diabetic Retinopathy (DR) is a major complication of diabetes and a leading cause of preventable blindness among working-age adults worldwide. Early detection and timely treatment are critical to preventing vision loss. However, mass screening is challenging due to:
- **A shortage of trained ophthalmologists** to interpret retinal scans.
- **The time-consuming nature** of manual diagnosis.
- **Lack of accessibility** to screening facilities in remote areas.

This creates a critical need for an automated, reliable, and accessible tool for preliminary DR screening.

---

## ✨ Our Solution

**AuraSight AI** addresses these challenges by providing an intelligent, web-based platform for the instant classification of Diabetic Retinopathy. A user can upload a retinal fundus image and receive a diagnosis of the DR severity stage within seconds.

Crucially, we tackle the "black box" problem of AI in medicine. By integrating **Explainable AI (XAI)**, our application generates a heatmap overlay on the original image, highlighting the specific regions (like microaneurysms or hemorrhages) that the model used to make its prediction. This builds trust and provides valuable insights for both patients and medical professionals.

---

## 🚀 Key Features

- **AI-Powered Diagnosis:** Classifies retinal scans into five stages of DR (No DR, Mild, Moderate, Severe, Proliferative) using an EfficientNetB0 model.
- **Explainable AI (XAI):** Generates Grad-CAM heatmaps to visualize the model's areas of focus, making the AI's decisions transparent.
- **Confidence Distribution Chart:** Displays a bar chart of the model's confidence across all five diagnostic categories for a richer analysis.
- **Professional PDF Reports:** Allows users to export a clean, professional PDF of the diagnosis, including images and results, for record-keeping.
- **Modern & Interactive UI:** Features a drag-and-drop file uploader and a gallery of example images for a seamless user experience.
- **Full-Stack Architecture:** Built with a scalable backend API and a responsive frontend, demonstrating a complete end-to-end development cycle.

---

## 🛠️ Tech Stack

| Category      | Technology                                           |
|---------------|------------------------------------------------------|
| **Frontend** | React.js, Axios, Chart.js, React-Dropzone            |
| **Backend** | Python, FastAPI, Uvicorn                             |
| **AI / ML** | TensorFlow, Keras, OpenCV, `tf-keras-vis` (for XAI) |
| **Deployment**| Vercel (Frontend), Render (Backend)                  |

---

##  perjalanan Project Journey & Challenges Faced

Building this application involved a complete machine learning and software development lifecycle. A key part of the journey was overcoming significant technical challenges, which demonstrates deep problem-solving skills:

1.  **TensorFlow Version & Environment Conflicts:** The most significant hurdle was a persistent series of model loading errors (`Shape Mismatch`, `AttributeError`) between the Google Colab training environment and the local Windows deployment environment. This was solved by:
    - Experimenting with multiple model saving formats (`.h5`, `SavedModel`, `.keras`).
    - Ultimately adopting a robust, industry-standard approach: rebuilding the model architecture in the backend and loading only the trained **weights**, which completely decoupled the application from the training environment and resolved all incompatibility issues.

2.  **Model Format for Keras 3:** The transition to Keras 3 introduced breaking changes, particularly the deprecation of `load_model()` for the legacy `SavedModel` format. This required adapting the code to use the new `TFSMLayer` for inference and later redesigning the entire model loading process to be format-agnostic.

3.  **Explainable AI Integration:** Implementing Grad-CAM with a `TFSMLayer` object proved incompatible. The solution was to load the model in two ways: a simple `TFSMLayer` for fast inference and a full `tf.keras.Model` object specifically for the XAI analysis, showcasing an ability to architect solutions for library-specific constraints.

---

## 🚀 Getting Started

To run this project locally, follow these steps:

### Prerequisites
- Git
- Python 3.10+
- Node.js and npm

### Backend Setup
1.  Clone the repository: `git clone https://github.com/YourUsername/AuraSight-AI-App.git`
2.  Navigate to the backend: `cd AuraSight-AI-App/backend`
3.  Create and activate a virtual environment:
    ```bash
    python -m venv env
    .\env\Scripts\activate
    ```
4.  Install dependencies: `pip install -r requirements.txt`
5.  **Important:** Place your trained model weights file (e.g., `aurasight_model.weights.h5`) in this directory.
6.  Run the server: `uvicorn main:app --reload`

### Frontend Setup
1.  In a new terminal, navigate to the frontend: `cd AuraSight-AI-App/frontend`
2.  Install dependencies: `npm install`
3.  Run the application: `npm start`
4.  Open `http://localhost:3000` in your browser.

---

## ⚠️ Disclaimer

AuraSight AI is a proof-of-concept project intended for educational and demonstrative purposes only. It is **not a certified medical device** and should not be used for actual medical diagnosis, treatment, or decision-making.
