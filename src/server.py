# src/server.py
import torch
import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import uvicorn

app = FastAPI()

# 1. Quantization: Shrinks the model to ~2GB so it fits easily on the T4 GPU.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

print("Loading model... This takes a minute.")
model_id = "google/gemma-3n-e2b-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="cuda", # Puts it on the T4 GPU
    trust_remote_code=True
)

def decode_image(base64_string):
    """Converts the text-encoded image back into a standard OpenCV image array."""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# 2. WebSocket Endpoint: An open tunnel for real-time data
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait for data from the frontend
            data = await websocket.receive_json()
            image_b64 = data.get("image")
            prompt = data.get("prompt", "What do you see?")
            
            if not image_b64:
                continue

            image = decode_image(image_b64)

            # 3. Process inputs (Handles both Hindi and English natively)
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
            
            # 4. Generate AI Response
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=50)
            
            response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Instantly send the text back through the tunnel
            await websocket.send_json({"response": response_text})

    except WebSocketDisconnect:
        print("Client disconnected.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)