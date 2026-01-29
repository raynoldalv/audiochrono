import streamlit as st
import librosa
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Raynold's Audio Chrono", layout="wide")

# --- FUNGSI FISIKA BALISTIK (MUZZLE VELOCITY) ---
def calculate_physics(dt_total, s, t_env, mass_gram):
    rho = 1.225      # kg/m3
    Cd = 0.47        # Sphere drag
    radius = 0.003   # 6mm BB (m)
    area = math.pi * (radius**2)
    m = mass_gram / 1000  # kg
    
    v_s = 331.4 * math.sqrt(1 + t_env / 273.15)
    t_sound_return = s / v_s
    t_flight = dt_total - t_sound_return
    
    if t_flight <= 0:
        return 0, 0, 0
    
    k = 0.5 * rho * Cd * area
    try:
        # Model Eksponensial Drag
        v0_m_s = (m * (math.exp(k * s / m) - 1)) / (k * t_flight)
        v0_fps = v0_m_s * 3.28084
        joules = 0.5 * m * (v0_m_s**2)
        v_avg_fps = (s / t_flight) * 3.28084
        return v0_fps, joules, v_avg_fps
    except:
        return 0, 0, 0

# --- SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- UI UTAMA ---
st.title("🎯 Raynold's Airsoft Audio Chronograph")
st.write("Gunakan menu 'Files' di HP untuk memilih rekaman audio.")

st.sidebar.header("⚙️ Parameter")
dist = st.sidebar.number_input("Jarak (m)", min_value=1.0, value=10.0, step=0.5)
massa = st.sidebar.selectbox("Massa BB (gram)", [0.12, 0.20, 0.25, 0.28, 0.30], index=1)
suhu = st.sidebar.slider("Suhu (°C)", 10, 40, 28)

# --- PERBAIKAN UPLOADER UNTUK MOBILE ---
uploaded_file = st.file_uploader("Unggah Rekaman", type=["wav", "mp3", "m4a", "aac"])

if uploaded_file is not None:
    with st.spinner('Menganalisis...'):
        # Librosa bisa membaca m4a/mp3 jika audioread terinstal
        y, sr = librosa.load(uploaded_file)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, wait=25)
        times = librosa.frames_to_time(onsets, sr=sr)
        
        if len(times) >= 2:
            dt_total = times[1] - times[0]
            v0, energy, v_avg = calculate_physics(dt_total, dist, suhu, massa)
            
            st.markdown("### 🚀 Hasil")
            c1, c2, c3 = st.columns(3)
            c1.metric("MUZZLE v0", f"{v0:.1f} FPS")
            c2.metric("ENERGY", f"{energy:.2f} J")
            c3.metric("AVG VEL", f"{v_avg:.1f} FPS")

            # Visualisasi
            fig, ax = plt.subplots(figsize=(10, 2.5))
            librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.5)
            ax.vlines([times[0], times[1]], -1, 1, color=['red', 'green'], linestyle='--')
            st.pyplot(fig)

            if st.button("➕ Simpan ke Riwayat"):
                st.session_state['history'].append({
                    "Waktu": datetime.now().strftime("%H:%M"),
                    "Massa": massa, "v0 (FPS)": round(v0, 1), "Joule": round(energy, 2)
                })
        else:
            st.error("Gunakan target nyaring agar audio terdeteksi di HP.")

# --- BATCH EXPORT ---
if st.session_state['history']:
    st.markdown("---")
    df = pd.DataFrame(st.session_state['history'])
    st.table(df)
    st.download_button("📥 Download CSV", df.to_csv(index=False), "riwayat_chrono.csv")
