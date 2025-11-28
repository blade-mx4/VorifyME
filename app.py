import io
import gc  # Garbage Collector to clean RAM
import numpy as np
import librosa
import tflite_runtime.interpreter as tflite  # Optimized for Render Free Tier
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

# ------------------------------------------------------------
# Initialize FastAPI app
# ------------------------------------------------------------
app = FastAPI(
    title="🎧 Deepfake Audio Detector API",
    description="An API for detecting AI-generated vs human voices using a CNN model (Lite Version).",
    version="1.0"
)

# ------------------------------------------------------------
# Enable CORS (so your frontend can talk to this API)
# ------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Load pre-trained TFLite model (Memory Efficient)
# ------------------------------------------------------------
print("🔁 Loading model...")

try:
    # Load the TFLite model instead of the heavy Keras model
    interpreter = tflite.Interpreter(model_path="model_v1.tflite")
    interpreter.allocate_tensors()

    # Get input and output details to use during prediction
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❌ Model Error: {e}")
    interpreter = None

# ------------------------------------------------------------
# Convert audio bytes -> Mel Spectrogram
# ------------------------------------------------------------
def audio_to_spectrogram(file_bytes: bytes):
    try:
        # 1. Safety: Limit input size immediately (10MB limit)
        if len(file_bytes) > 10 * 1024 * 1024:
            print("⚠️ File too large")
            return None
            
        # Load the audio file into memory
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        
        # 2. Memory Trick: Export only the first 4 second 
        
        # Export to WAV in memory
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # 3. Load with duration limit (Double safety)
        y, sr = librosa.load(wav_io, sr=None, duration=4.0)
        
        # Generate Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        
        # Resize to match model input shape
        S_DB = np.resize(S_DB, (128, 128))
        
        # Normalize between 0–1
        S_DB = (S_DB - S_DB.min()) / (S_DB.max() - S_DB.min())
        
        # Force clean up memory immediately
        del audio, wav_io, y, S
        gc.collect()
        
        return S_DB
    except Exception as e:
        print(f"⚠️ Error processing file: {e}")
        return None


# ------------------------------------------------------------
# API route for predictions
# ------------------------------------------------------------
@app.get("/")
def home_get():
    return {"message": "Server is running"}  

@app.head("/")
def home_head():
    return Response(status_code=200)   

@app.head("/health")
def health():
    return Response(status_code=200)

@app.post("/audio")
def predict_audio(file: UploadFile = File(...)):
    if interpreter is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server."})
    
    try:
        # Read the uploaded audio file
        file_bytes = file.file.read()
        
        # Convert to mel spectrogram
        spectrogram = audio_to_spectrogram(file_bytes)
        if spectrogram is None:
            return JSONResponse(status_code=400, content={"error": "Could not process audio file."})
        
        # Add batch and channel dimensions for CNN input (1, 128, 128, 1)
        # Note: We use float32 as that is standard for TFLite inputs
        input_data = spectrogram[np.newaxis, ..., np.newaxis].astype(np.float32)
        
        # Run prediction using TFLite Interpreter
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get the result
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred = output_data[0][0]
        
        print("F✅ Prediction successful.")
        
        # Interpret result
        if pred < 0.5:
            label = "HUMAN"
            confidence = (1 - pred) * 100
        else:
            label = "AI"
            confidence = pred * 100
            
        print("Classification:", label, f"({confidence:.2f}%)")
        
        # Force clean memory
        del file_bytes, spectrogram, input_data
        gc.collect()
        
        return JSONResponse(content={
            "classification": label,
            #"confidence": round(float(confidence), 2)
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------------
# Root endpoint
# ------------------------------------------------------------
@app.get("/health")
def root():
    return {"message": "🎧 Deepfake Audio Detector API is running!"}


# ------------------------------------------------------------
# Run server (if run directly)
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import os
    print("🚀 Starting API server...")
    port = 10000
    uvicorn.run(app, host="0.0.0.0", port=port)
