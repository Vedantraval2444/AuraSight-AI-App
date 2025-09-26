import uvicorn
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import tensorflow as tf
import io
import base64
from fpdf import FPDF
from datetime import datetime

# XAI Imports
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# --- 1. INITIALIZE THE APP ---
app = FastAPI(title="AuraSight AI Diagnostics")
origins = [
    "http://localhost:3000",  # For local development
    "https://aurasight24.onrender.com",  # Your live frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- 2. REBUILD MODEL STRUCTURE AND LOAD WEIGHTS ---
IMG_SIZE = 224
NUM_CLASSES = 5

def build_model():
    base_model = tf.keras.applications.EfficientNetB0(
        weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    predictions = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model

try:
    model = build_model()
    model.load_weights('aurasight_model.weights.h5')
    print("Model structure built and weights loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# --- 3. XAI: GENERATE GRAD-CAM HEATMAP ---
def generate_gradcam(image_batch, prediction_index):
    score = CategoricalScore([prediction_index])
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    heatmap = gradcam(score, image_batch, penultimate_layer=-1)
    heatmap = (heatmap[0] * 255).astype(np.uint8)
    original_image = (image_batch[0] * 255).astype(np.uint8)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_img, 0.4, original_image, 0.6, 0)
    is_success, buffer = cv2.imencode(".jpg", superimposed_img)
    return base64.b64encode(buffer).decode("utf-8")

# --- 4. PREDICTION FUNCTION ---
def preprocess_and_predict(image_bytes: bytes):
    if model is None: return None
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    prediction_array = model.predict(img_batch)[0]
    predicted_index = np.argmax(prediction_array)
    heatmap_base64 = generate_gradcam(img_batch, predicted_index)
    return prediction_array, heatmap_base64

# --- 5. PDF GENERATION ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AuraSight AI Diagnostic Report', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report(data: dict):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.cell(0, 10, f"Patient File: {data['filename']}", 0, 1)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Diagnosis Result', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Condition: {data['diagnosis']}", 0, 1)
    pdf.cell(0, 10, f"Confidence: {data['confidence']}", 0, 1)
    pdf.ln(10)
    for title, b64_string in [("Original Scan", data['original_image']), ("Model Focus Heatmap", data['heatmap_image'])]:
        try:
            image_bytes = base64.b64decode(b64_string)
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, title, 0, 1)
            pdf.image(io.BytesIO(image_bytes), w=150)
            pdf.ln(10)
        except Exception as e:
            print(f"Could not process image for PDF: {e}")
    
    # --- THIS IS THE FIX ---
    # The .output() method directly returns bytes, so no .encode() is needed
    return pdf.output()

# --- 6. API ENDPOINTS ---
@app.get("/")
def read_root(): return {"message": "Welcome to the AuraSight AI API"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = preprocess_and_predict(image_bytes)
    if result is None:
        return {"error": "Model not loaded or prediction failed"}
    prediction_array, heatmap = result
    predicted_index = np.argmax(prediction_array)
    confidence = prediction_array[predicted_index] * 100
    return {
        "filename": file.filename,
        "diagnosis": CLASS_NAMES[predicted_index],
        "confidence": f"{confidence:.2f}%",
        "probabilities": (prediction_array * 100).tolist(),
        "heatmap_image": heatmap
    }

@app.post("/export_pdf")
async def export_pdf(data: dict = Body(...)):
    pdf_bytes = create_pdf_report(data)
    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers={"Content-Disposition": "attachment;filename=AuraSight_Report.pdf"})