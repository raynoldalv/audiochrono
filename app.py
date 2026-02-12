import streamlit as st
import librosa
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import os
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Raynold's Audio Chrono Pro", layout="wide")

# --- FUNGSI FISIKA BALISTIK ---
def calculate_physics(dt_total, s, t_env, mass_gram):
    rho = 1.225
    Cd = 0.47
    radius = 0.003
    area = math.pi * (radius**2)
    m = mass_gram / 1000 
    
    v_s = 331.4 * math.sqrt(1 + t_env / 273.15)
    t_sound_return = s / v_s
    t_flight = dt_total - t_sound_return
    
    if t_flight <= 0.01: # Filter jika waktu terlalu singkat (error deteksi)
        return 0, 0, 0
    
    k = 0.5 * rho * Cd * area
    try:
        v0_m_s = (m * (math.exp(k * s / m) - 1)) / (k * t_flight)
        v0_fps = v0_m_s * 3.28084
        joules = 0.5 * m * (v0_m_s**2)
        v_avg_fps = (s / t_flight) * 3.28084
        return v0_fps, joules, v_avg_fps
    except:
        return 0, 0, 0

# --- UI UTAMA ---
st.title("Audiochrono")
st.markdown("Analisis otomatis untuk rekaman tunggal maupun beruntun (Batch).")

# Sidebar
st.sidebar.header("⚙️ Konfigurasi")
dist = st.sidebar.number_input("Jarak ke Target (m)", min_value=1.0, value=5.0, step=0.5)
massa = st.sidebar.selectbox("Massa BB (gram)", [0.12, 0.20, 0.25, 0.28, 0.30], index=1)
suhu = st.sidebar.slider("Suhu Lokasi (°C)", 10, 40, 28)
st.sidebar.markdown("---")
sensitivitas = st.sidebar.slider("Sensitivitas Deteksi", 0.01, 0.20, 0.07, step=0.01, help="Kecilkan jika tembakan tidak terdeteksi.")

st.sidebar.write("Project by: **Raynold - FMIPA ITB**")

# File Uploader (Mobile Friendly)
uploaded_file = st.file_uploader("Unggah Rekaman (WAV, MP3, M4A)", type=None)

if uploaded_file is not None:
    # Simpan ke file sementara (Penting untuk format m4a/mobile)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        with st.spinner('Mendeskripsi gelombang suara...'):
            y, sr = librosa.load(tmp_path)
            # Deteksi Onset (Menggunakan parameter sensitivitas dari sidebar)
            onsets = librosa.onset.onset_detect(y=y, sr=sr, wait=15, pre_avg=15, post_avg=15, delta=sensitivitas)
            times = librosa.frames_to_time(onsets, sr=sr)
            
            if len(times) >= 2:
                results = []
                # Pasangkan Onset (Tembakan & Hantaman) secara berurutan
                # Melompat per 2 puncak (0&1, 2&3, dst)
                for i in range(0, len(times) - 1, 2):
                    t_shot = times[i]
                    t_impact = times[i+1]
                    dt = t_impact - t_shot
                    
                    # Validasi: Airsoft biasanya butuh 0.05 - 0.5 detik untuk jarak menengah
                    if 0.02 < dt < 1.0:
                        v0, j, vavg = calculate_physics(dt, dist, suhu, massa)
                        if v0 > 50: # Filter noise
                            results.append({
                                "Shot #": (i // 2) + 1,
                                "v0 (FPS)": round(v0, 1),
                                "Energy (J)": round(j, 2),
                                "Avg (FPS)": round(vavg, 1),
                                "Time (s)": round(dt, 3)
                            })

                if results:
                    df = pd.DataFrame(results)
                    
                    # Tampilan Summary
                    st.success(f"Berhasil mendeteksi {len(results)} tembakan!")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.table(df)
                    with col2:
                        avg_v0 = df["v0 (FPS)"].mean()
                        st.metric("RATA-RATA FPS", f"{avg_v0:.1f}")
                        st.metric("KONSISTENSI (STD)", f"{df['v0 (FPS)'].std():.1f} FPS")

                    # Visualisasi Waveform
                    fig, ax = plt.subplots(figsize=(12, 3))
                    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.4)
                    ax.vlines(times, -1, 1, color='red', linestyle='--', alpha=0.6)
                    ax.set_title("Analisis Puncak Suara (Merah = Deteksi)")
                    st.pyplot(fig)
                    
                    # Download
                    st.download_button("📥 Export Hasil ke CSV", df.to_csv(index=False), "chrono_batch.csv")
                else:
                    st.error("Deteksi gagal. Pastikan jeda antara tembakan dan hantaman jelas.")
            else:
                st.warning("Hanya satu puncak terdeteksi. Gunakan target yang lebih nyaring.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

st.info("Tips: Untuk 10 tembakan, pastikan ada jeda minimal 1 detik antar tembakan agar tidak tumpang tindih.")
