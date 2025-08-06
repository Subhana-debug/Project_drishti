# ------------------ Required Libraries ------------------
import streamlit as st
import cv2
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from deepface import DeepFace
import sounddevice as sd
import soundfile as sf
import librosa
import tempfile
import folium
import shutil
import requests
from streamlit_folium import st_folium
import pandas as pd
    


# ------------------ Initialization ------------------
st.set_page_config(page_title="Project Drishti", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None

# ------------------ Helper Functions ------------------
def authenticate(username, password, user_type):
    users = {"admin": "admin123", "user": "user123"}
    return users.get(username) == password

def capture_and_save_image(zone_name):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        folder = f"captured_frames/{zone_name}"
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"frame_{ts}.jpg")
        cv2.imwrite(path, frame)
    cap.release()


# ------------------ Login Page ------------------



if not st.session_state.logged_in:
    st.title("🔐 Project Drishti Login")

    st.markdown("""
    <div style='text-align:center; padding:20px'>
        <h2 style='color:#0073e6;'>Transforming <b>reactive monitoring</b> into <b>proactive prevention</b>.</h2>
        <h4 style='color:#333;'>Let's make every public event <b style='color:#e60000;'>safer</b>, 
        <b style='color:#009933;'>smarter</b>, and <b style='color:#cc00cc;'>stress-free</b>.</h4>
    </div>
    """, unsafe_allow_html=True)

    with st.form("Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        user_type = st.radio("Login as", ["commander", "user"])
        if st.form_submit_button("Login"):
            if authenticate(username, password, user_type):
                st.session_state.logged_in = True
                st.session_state.role = user_type
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()
# Sidebar Navigation FIRST (this must be before the if selected == ...)
st.sidebar.title("📋 Navigation")

# Role-based menu (make sure this is placed above all `if selected == ...`)
if st.session_state.role == "user":
    menu = ["Home", "Live Map", "Lost & Found",  "AI Forecasting,Gemini Chat","Feedback", "Developers"]
else:
    menu = ["Home","Webcam Monitor", "Live Map", "Lost & Found", "AI Forecasting","Gemini Chat", "Crowd's Emotion", "Sound Detection", "Feedback", "Developers"]

# Define selected page (this is the MOST important line before using 'selected')
selected = st.sidebar.radio("📂 Navigate", menu)




if selected == "Home":
    st.markdown(
        """
        <div style='text-align:center; padding:30px 10px; background-color:#e6f2ff; border-radius:10px'>
            <h2 style='color:#0073e6;'>Transforming <b>reactive monitoring</b> into <b>proactive prevention</b>.</h2>
            <h4 style='color:#333;'>Let's make every public event <b style='color:#e60000;'>safer</b>, 
            <b style='color:#009933;'>smarter</b>, and <b style='color:#cc00cc;'>stress-free</b>.</h4>
            <p style='font-size:20px; color:#444; margin-top:20px;'><i>“Every face, every sound — Drishti sees and hears it all.”</i></p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("---")

    # App Description
    st.markdown(
        """
        <div style='background-color: #ffe6f0; padding: 25px; border-radius: 15px; border: 2px solid #ffb3cc;'>
            <h2 style='color: #cc0066; text-align: center;'>👁️ Welcome to <b>Project Drishti</b> – where AI becomes your event’s <span style="color:#ff6699;">sixth sense</span>.</h2>
            <p style='font-size: 18px; color: #4d004d; text-align: center;'>
                We <b style='color:#cc0066;'>watch the crowd</b>, <b style='color:#ff3385;'>detect the pulse</b>, and <b style='color:#9900cc;'>alert before chaos begins</b>.<br><br>
                From <span style="color:#0099cc;"><b>emotions</b></span> to <span style="color:#cc0000;"><b>emergencies</b></span>, everything is <b>seen</b>, <b>heard</b>, and <b>handled</b>.
            </p>
            <h4 style='text-align:center; color:#e60000;'>Because <i>safety shouldn't wait</i> for something to go wrong.</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("📩 Contact Us"):
            st.markdown("## 📞 Contact Us")
            st.markdown("""
                **Commander Hotline:** +91 98765 43210  
                **Support Email:** drishti.command@secureai.com  
                **Event Operations Office:** Room 208, Control Block  
            """)

    with col2:
        if st.button("👥 Meet the Team"):
            zone_selected = st.selectbox("Which Zone are you in?", ["Zone A", "Zone B", "Zone C", "Unknown"])
            if zone_selected == "Zone A":
                st.markdown("### 👨‍💼 Zone A Team")
                st.write("• Officer Ravi Kumar – Surveillance Lead")
                st.write("• Volunteer Anitha – Emergency Support")
            elif zone_selected == "Zone B":
                st.markdown("### 👩‍💼 Zone B Team")
                st.write("• Commander Priya Rajan – Control Lead")
                st.write("• Volunteer Jayanth – Safety Operations")
            elif zone_selected == "Zone C":
                st.markdown("### 👨‍🚒 Zone C Team")
                st.write("• Inspector Arjun Das – Crowd Specialist")
                st.write("• Volunteer Meena – Public Support")
            else:
                st.warning("No team assigned. Please contact control center.")

    with col3:
        st.empty()



# ------------------ Webcam Monitor ------------------
elif selected == "Webcam Monitor":


    # Set up directory
    os.makedirs("captured_frames", exist_ok=True)

    st.header("📷 Webcam Monitoring - Zone Wise")
    zone = st.selectbox("Select Zone", ["Zone A", "Zone B", "Zone C"])

    # Start/Stop button states
    if 'run_webcam' not in st.session_state:
        st.session_state.run_webcam = False

    start_btn = st.button("✅ Start Webcam")
    stop_btn = st.button("🛑 Stop Webcam")

    if start_btn:
        st.session_state.run_webcam = True
    if stop_btn:
        st.session_state.run_webcam = False

    FRAME_WINDOW = st.image([])

    # Start capturing
    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        last_capture_time = time.time()

        while st.session_state.run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Failed to access webcam.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

            # Save frame every 5 seconds
            current_time = time.time()
            if current_time - last_capture_time > 5:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                folder_path = os.path.join("captured_frames", zone)
                os.makedirs(folder_path, exist_ok=True)

                filename = f"{folder_path}/frame_{timestamp}.jpg"
                Image.fromarray(frame).save(filename)
                st.success(f"📸 Captured at {timestamp} in {zone}")
                last_capture_time = current_time

            # Exit loop if stop pressed
            if stop_btn:
                break

        cap.release()
        st.session_state.run_webcam = False
    else:
        st.info("Start the webcam to begin monitoring.")


# ------------------ Lost & Found ------------------

elif selected == "Lost & Found":
    # ------------- Windows Fix for OpenCV / DeepFace -------------
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

    # ------------- Directory Setup -------------
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("captured_frames", exist_ok=True)  # This should contain subfolders like Zone A, Zone B, etc.

    # ------------- UI -------------
    
    st.title("🔍 Lost & Found - Face Recognition")
    uploaded_file = st.file_uploader("Upload photo of the lost person", type=["jpg", "jpeg", "png"])

    # ------------- Function Helpers -------------
    def get_zone_from_path(path):
        parts = path.replace("\\", "/").split("/")
        for part in parts:
            if "zone" in part.lower():
                return part
        return "Unknown Zone"

    def get_time_from_filename(path):
        try:
            filename = os.path.basename(path)
            parts = filename.split("_")
            timestamp = parts[1] + parts[2].split(".")[0]
            dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
            return dt.strftime("%d %B %Y, %I:%M:%S %p")
        except:
            return "Unknown Time"

    # ------------- Lost & Found Logic -------------
    if uploaded_file:
        img_path = os.path.join("uploads", "lost_person.jpg")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(img_path, caption="Uploaded image for matching", width=300)

        st.info("🔍 Searching for matches...")

        match_found = False

        for root, dirs, files in os.walk("captured_frames"):
            for file in files:
                if file.endswith((".jpg", ".png", ".jpeg")):
                    try:
                        full_path = os.path.join(root, file)
                        result = DeepFace.verify(
                            img1_path=img_path,
                            img2_path=full_path,
                            model_name="VGG-Face",
                            detector_backend="retinaface",
                            enforce_detection=False
                        )
                        if result["verified"]:
                            st.success("✅ Match found!")
                            st.image(full_path, caption="Matched Frame", width=300)

                            zone = get_zone_from_path(full_path)
                            timestamp = get_time_from_filename(full_path)

                            st.info(f"📍 Last seen in: **{zone}**")
                            st.info(f"🕒 Time: **{timestamp}**")
                            match_found = True
                            break
                    except Exception as e:
                        st.warning(f"⚠ Error comparing with {file}: {e}")

            if match_found:
                break

        if not match_found:
            st.error("❌ No match found in captured frames.")

# ------------------ Crowd's Emotion ------------------
elif selected == "Crowd's Emotion":
    st.header("😊 Crowd Emotion Estimation via Webcam")

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    st.info("🎥 Analyzing emotions. Press 'Stop' on sidebar to quit.")

    emotion_count = {
        'angry': 0,
        'disgust': 0,
        'fear': 0,
        'happy': 0,
        'sad': 0,
        'surprise': 0,
        'neutral': 0
    }

    run = st.checkbox("Start Camera")

    if run:
        start_time = time.time()
        duration = 30  # seconds to analyze
        with st.spinner("Analyzing crowd emotion for 30 seconds..."):
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Webcam not found.")
                    break

                try:
                    analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    emotion = analysis[0]['dominant_emotion']
                    emotion_count[emotion] += 1

                    # Display live feed
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.putText(frame, f"Emotion: {emotion}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    FRAME_WINDOW.image(frame)

                except Exception as e:
                    st.error(f"DeepFace error: {e}")
                    break

        cap.release()
        FRAME_WINDOW.empty()

        st.success("✅ Emotion capture completed!")

        # Display bar chart with highlight
        st.subheader("🔎 Emotion Distribution")
        max_emotion = max(emotion_count, key=emotion_count.get)

        for emotion, count in emotion_count.items():
            color = "green" if emotion == max_emotion else "gray"
            st.markdown(
                f"<div style='margin-bottom:8px; color:{color}; font-size:18px;'>"
                f"<b>{emotion.capitalize()}</b>: {count} frames</div>",
                unsafe_allow_html=True
            )
    else:
        cap.release()
        st.info("☝️ Click the checkbox to start camera.")

# ------------------ Sound Detection ------------------
elif selected == "Sound Detection":
    st.header("🎤 Real-Time Sound Analysis")

    def record_audio(duration, sample_rate):
        st.info(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        return audio

    def detect_loudness(y):
        rms = np.sqrt(np.mean(y**2))
        return "High" if rms > 0.05 else "Normal"

    def detect_emotion(y):
        return "Angry" if np.mean(y) > 0.01 else "Neutral"

    if st.button("🎧 Record Audio"):
        audio = record_audio(5, 22050)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, audio, 22050)
            tmp_path = tmpfile.name

        st.audio(tmp_path, format="audio/wav")

        y, sr = librosa.load(tmp_path, sr=None)

        loudness = detect_loudness(y)
        emotion = detect_emotion(y)

        color_emotion = "red" if emotion == "Angry" else "green"
        color_loud = "orange" if loudness == "High" else "blue"

        st.markdown(
            f"🔊 Loudness: <span style='color:{color_loud}; font-weight:bold;'>{loudness}</span><br>"
            f"😡 Emotion: <span style='color:{color_emotion}; font-weight:bold;'>{emotion}</span>",
            unsafe_allow_html=True
        )

        if loudness == "High" and emotion == "Angry":
            st.error("⚠ Fight likely! Send a volunteer immediately.")
        elif emotion == "Angry":
            st.warning("😐 Frustration detected. Monitor the area.")
        else:
            st.success("✅ Environment seems calm.")

        os.remove(tmp_path)


# ------------------ Live Map ------------------
elif selected == "Live Map":
    st.title("🗺 Live Crowd Map")
    def get_location():
        try:
            loc = requests.get("http://ip-api.com/json").json()
            return loc['lat'], loc['lon']
        except:
            return 12.9784, 77.5996

    lat, lon = get_location()
    zones = [
        {"name": "Zone A", "lat": lat+0.0005, "lon": lon-0.0004, "count": 9},
        {"name": "Zone B", "lat": lat-0.0006, "lon": lon+0.0003, "count": 28},
        {"name": "Zone C", "lat": lat, "lon": lon+0.0006, "count": 40},
    ]
    m = folium.Map(location=[lat, lon], zoom_start=17)
    for z in zones:
        folium.CircleMarker(
            location=[z['lat'], z['lon']],
            radius=18,
            color="red" if z['count'] > 30 else "orange" if z['count'] > 10 else "green",
            popup=f"{z['name']}: {z['count']} People\n({z['lat']}, {z['lon']})",
            fill=True
        ).add_to(m)

    st_folium(m, width=800, height=500)

elif selected == "Feedback":
  
    st.header("🗣 Anonymous Feedback Zone")

    st.markdown(
        """
        <p style="font-size:16px;">
        Your feedback helps <b style='color:#0066cc;'>volunteers</b> and <b style='color:#cc0000;'>officers</b> act quickly.
        Please share how you feel — <b style='color:green;'>no identity is stored</b>.
        </p>
        """, unsafe_allow_html=True
    )

    # CSV file to store feedback
    FEEDBACK_FILE = "user_feedback.csv"

    # Feedback options
    feedback_options = [
        "😟 I feel uncomfortable",
        "👥 It’s too crowded here",
        "🔊 The noise is disturbing",
        "💬 Other (I’ll type it)"
    ]

    selected_option = st.radio("Select your concern:", feedback_options)

    custom_feedback = ""
    if selected_option == "💬 Other (I’ll type it)":
        custom_feedback = st.text_area("Type your feedback:")

    if st.button("✅ Submit Feedback"):
        # Prepare data
        feedback_text = custom_feedback if custom_feedback else selected_option
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to CSV
        feedback_entry = pd.DataFrame([[timestamp, feedback_text]], columns=["Time", "Feedback"])
        if os.path.exists(FEEDBACK_FILE):
            feedback_entry.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        else:
            feedback_entry.to_csv(FEEDBACK_FILE, index=False)
        
        st.success("🎉 Feedback submitted anonymously. Thank you for helping us improve!")

    # Admin view (optional)
    with st.expander("📋 Admin View: Show recent feedback log"):
        if os.path.exists(FEEDBACK_FILE):
            data = pd.read_csv(FEEDBACK_FILE)
            st.dataframe(data.tail(10))
        else:
            st.warning("📭 No feedback submitted yet.")

    #--------AI Forecasting---------

    # ------------------ AI Forecasting ------------------
elif selected == "AI Forecasting":
    st.header("📈 AI-Based Crowd Forecasting")
    st.markdown("Upload the **Vertex AI Forecasting CSV** to visualize predicted crowd surges over time.")

    uploaded_csv = st.file_uploader("📂 Upload Forecasting Output (CSV)", type=["csv"])

    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)

            # Auto-detect timestamp column
            time_col = next((col for col in df.columns if "time" in col.lower()), None)
            if not time_col:
                time_col = st.selectbox("Select Timestamp Column", df.columns)
            else:
                st.success(f"✅ Detected Timestamp Column: `{time_col}`")

            # Auto-detect crowd prediction column
            crowd_col = next((col for col in df.columns if "crowd" in col.lower()), None)
            if not crowd_col:
                crowd_col = st.selectbox("Select Crowd Prediction Column", df.columns)
            else:
                st.success(f"✅ Detected Prediction Column: `{crowd_col}`")

            # Parse and plot
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col, crowd_col])

            st.line_chart(df.set_index(time_col)[crowd_col])
            st.success("🎯 Forecast Visualized Successfully!")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.info("📌 Please upload your CSV file to begin.")





# ---------------------------- Gemini Chat Page ----------------------------
elif selected == "Gemini Chat":
    st.header("💬 Ask Drishti (Gemini AI Chatbot)")
    # ---------------------------- CONFIG ----------------------------
    API_KEY = "AIzaSyBIOGc6eTrUcPkVYaDwXK50aVP9M-KMvBs"
    API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent?key={API_KEY}"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Type your question here...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Prepare the prompt in Gemini format
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": user_input}]}
            ]
        }

        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()

            if "candidates" in result and result["candidates"]:
                reply = result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                reply = "⚠️ No response from Gemini."

        except Exception as e:
            reply = f"❌ Error: {e}"

        st.chat_message("assistant").markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # ------------------ Developers Page ------------------
elif selected == "Developers":
    st.title("👩‍💻 Meet the Developers")

    col1, col2 = st.columns(2)

    with col1:
        st.image("subhana.jpg", width=200)
        st.markdown("""
        <h3 style='color:#e60073;'>Subhana K</h3>
        <p style='font-size:16px;'>🎨 Frontend Developer</p>
        """, unsafe_allow_html=True)

    with col2:
        st.image("Sri Harini.jpg", width=200)
        st.markdown("""
        <h3 style='color:#3366cc;'>Sri Harini S</h3>
        <p style='font-size:16px;'>🧠 Backend Developer</p>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("We built this project with love and lots of debugging. 💻🛠️")






