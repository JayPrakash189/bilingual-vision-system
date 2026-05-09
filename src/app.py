# src/app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import asyncio
import websockets
import json
import base64
import cv2
from threading import Thread

st.set_page_config(page_title="Bilingual Vision AI")
st.title("Real-Time Bilingual AI Camera")

# This URL will change when you deploy to Lightning AI
WS_URL = st.secrets.get("BACKEND_URL", "ws://localhost:8000/ws")

if 'latest_frame' not in st.session_state:
    st.session_state.latest_frame = None

def video_frame_callback(frame):
    """Grabs the video frame from your webcam."""
    img = frame.to_ndarray(format="bgr24")
    st.session_state.latest_frame = img
    return frame

# Start the webcam
ctx = webrtc_streamer(
    key="gemma-vision",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)

user_prompt = st.text_input("Prompt (Hindi/English):", "Describe this scene.")
output_text = st.empty()

# Background task to send frames to the GPU
async def stream_to_gpu():
    try:
        async with websockets.connect(WS_URL) as websocket:
            while ctx.state.playing:
                if st.session_state.latest_frame is not None:
                    # Compress frame to send over the internet
                    _, buffer = cv2.imencode('.jpg', st.session_state.latest_frame)
                    img_b64 = base64.b64encode(buffer).decode('utf-8')

                    # Send to Lightning AI
                    await websocket.send(json.dumps({
                        "image": img_b64,
                        "prompt": user_prompt
                    }))

                    # Get response back
                    response = await websocket.recv()
                    data = json.loads(response)
                    output_text.markdown(f"**AI:** {data['response']}")
                
                # Wait 1 second before sending the next frame so the GPU doesn't crash
                await asyncio.sleep(1)
    except Exception as e:
        pass # Handle disconnection gracefully

# Run the background task without freezing the website
if ctx.state.playing:
    Thread(target=lambda: asyncio.run(stream_to_gpu()), daemon=True).start()