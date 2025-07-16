import os
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import joblib
from fpdf import FPDF
from datetime import datetime

# Load trained model
model = joblib.load("depression_model.pkl")

# 🎨 Page setup
st.set_page_config(page_title="Child Depression Detection", page_icon="🎙️", layout="centered")
# 🌸 Background and font color fix
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #ffe6f0, #fffde6);
            color: black !important;
        }

        h1, h2, h3, h4, h5, h6, p, label, .markdown-text-container {
            color: black !important;
        }

        .css-1cpxqw2, .css-ffhzg2, .css-1v0mbdj, .css-10trblm, .css-qrbaxs {
            color: black !important;
        }

        .stMarkdown, .stText, .stTextInput, .stSelectbox, .stSlider {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)


# 🌸 Background and heading color
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #ffe6f0, #fffde6);
        }
        h1 span {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# 🔷 Logo
st.image("logo.png", width=120)

# 🔥 Title
st.markdown("""
    <div style='text-align: center;'>
        <h1><span>🎙️ Child Speech Depression Detection</span></h1>
        <p style='font-size:18px;'>Upload a .wav file and find out if it shows signs of depression</p>
    </div>
""", unsafe_allow_html=True)

# 📚 Sidebar instructions
st.sidebar.header("🔧 Instructions")
st.sidebar.markdown("""
1. Upload a **short .wav** file (3-5 seconds).  
2. Use clean audio with no background noise.  
3. See prediction, graphs, and confidence score!
""")

# 🗂️ Store session history
if "history" not in st.session_state:
    st.session_state.history = []

# 📤 File uploader
uploaded_file = st.file_uploader("📁 Upload your .wav file here:", type=["wav"])

# 🎵 Feature extractor
def extract_features(file):
    try:
        y, sr = librosa.load(file, sr=None, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc.mean(axis=1).reshape(1, -1), y, sr
    except Exception as e:
        st.error(f"Audio Error: {e}")
        return None, None, None

# 📄 PDF Report
def generate_pdf_report(prediction_label, confidence):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    advice = "Please consult a professional if this is a real concern." if prediction_label == "Depressed" else "Voice sounds healthy. Keep monitoring regularly."

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Child Speech Depression Detection Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date & Time: {now}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction Result: {prediction_label}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Score: {confidence}%", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Note:\n{advice}")

    filename = "depression_report.pdf"
    pdf.output(filename)
    return filename

# 🔍 Prediction Block
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    features, y, sr = extract_features(uploaded_file)

    if features is not None:
        # 🎼 Waveform
        st.markdown("### 🎵 Voice Waveform")
        fig, ax = plt.subplots()
        ax.plot(y)
        ax.set_title("Waveform")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        # 🎛️ Mel Spectrogram
        st.markdown("### 🔊 Mel Spectrogram")
        spec = librosa.feature.melspectrogram(y=y, sr=sr)
        fig2, ax2 = plt.subplots()
        img = librosa.display.specshow(librosa.power_to_db(spec, ref=np.max), sr=sr, x_axis='time', y_axis='mel', ax=ax2)
        ax2.set(title='Mel Spectrogram')
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        st.pyplot(fig2)

        # 🤖 Predict
        prediction = model.predict(features)
        proba = model.predict_proba(features)[0]
        confidence_score = round(proba[1] * 100, 2) if len(proba) > 1 else round(proba[0] * 100, 2)
        result_label = "Depressed" if prediction[0] == 1 else "Not Depressed"

        # 🎯 Result Display
        if prediction[0] == 1:
            st.markdown(f"""
                <div style="background-color: #ffe6e6; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: red;">⚠️ Depression Detected</h2>
                    <p>Please consult a professional if this is a real concern.</p>
                    <p><b>Confidence:</b> {confidence_score}%</p>
                </div>
            """, unsafe_allow_html=True)
            st.session_state.history.append(f"Depressed ({confidence_score}%)")
        else:
            st.markdown(f"""
                <div style="background-color: #e6fff5; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: green;">✅ No Depression Detected</h2>
                    <p>The voice sounds healthy and normal.</p>
                    <p><b>Confidence:</b> {100 - confidence_score}%</p>
                </div>
            """, unsafe_allow_html=True)
            st.session_state.history.append(f"Not Depressed ({100 - confidence_score}%)")

        # 📄 Download Report
        report_path = generate_pdf_report(result_label, confidence_score)
        with open(report_path, "rb") as file:
            st.download_button("📄 Download Report as PDF", file, file_name="depression_report.pdf", mime="application/pdf")

        # 📜 Prediction History
        st.markdown("### 📜 Prediction History (This Session)")
        for i, entry in enumerate(st.session_state.history[::-1]):
            st.write(f"{i+1}. {entry}")

        # 🧸 Teddy Character Assistant
        st.markdown("### 🐻 Teddy's Advice")
        if prediction[0] == 1:
            st.image("sad_teddy.png", width=120)
            st.markdown("""
                <div style='background-color: #fff3f3; padding: 15px; border-radius: 10px;'>
                    <b>Teddy says:</b><br>
                    😔 "You seem a bit down today... Don't worry, you're not alone."<br>
                    🧸 "Talk to someone you love, okay?"
                </div>
            """, unsafe_allow_html=True)
        else:
            st.image("happy_teddy.png", width=120)
            st.markdown("""
                <div style='background-color: #f0fff4; padding: 15px; border-radius: 10px;'>
                    <b>Teddy says:</b><br>
                    😊 "You sound cheerful today! Keep smiling!"<br>
                    🎉 "I'm proud of you!"
                </div>
            """, unsafe_allow_html=True)

        # 🌟 Mood Rating
        st.markdown("### 🌟 How Are You Feeling Today?")
        rating = st.slider("Rate your mood from 1 (low) to 5 (great)", 1, 5)

        if rating == 5:
            st.success("🥳 You're shining bright today! Keep it up!")
            st.balloons()
        elif rating == 4:
            st.info("😊 You seem to be in a good mood! Awesome!")
            st.balloons()
        elif rating == 3:
            st.warning("😌 It's okay to feel neutral. Maybe do something you love?")
        elif rating == 2:
            st.error("😕 Hang in there. Talk to someone you trust 💙")
            st.snow()
        else:
            st.error("😢 It's okay to feel sad. You're never alone 💖")
            st.snow()
