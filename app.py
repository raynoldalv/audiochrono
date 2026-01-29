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
    """
    Menghitung Muzzle Velocity (v0) dengan kompensasi hambatan udara (drag).
    """
    rho = 1.225      # Massa jenis udara (kg/m3)
    Cd = 0.47        # Koefisien drag bola (Sphere)
    radius = 0.003   # Radius BB 6mm (m)
    area = math.pi * (radius**2)
    m = mass_gram / 1000  # Konversi ke kg
    
    # 1. Kecepatan Suara (v_s) berdasarkan suhu Celsius
    v_s = 331.4 * math.sqrt(1 + t_env / 273.15)
    
    # 2. Waktu rambat suara kembali dari target ke mikrofon
    t_sound_return = s / v_s
    t_flight = dt_total - t_sound_return
    
    if t_flight <= 0:
        return 0, 0, 0
    
    # 3. Perhitungan Muzzle Velocity (v0) menggunakan model Drag
    # k = konstanta drag balistik
    k = 0.5 * rho * Cd * area
    
    try:
        # Rumus Integrasi Drag: v0 = (m * (exp(ks/m) - 1)) / (kt)
        v0_m_s = (m * (math.exp(k * s / m) - 1)) / (k * t_flight)
        v0_fps = v0_m_s * 3.28084
        
        # 4. Energi Kinetik (Joule)
        joules = 0.5 * m * (v0_m_s**2)
        
        # 5. Kecepatan Rata-rata (Avg FPS)
        v_avg_fps = (s / t_flight) * 3.28084
        
        return v0_fps, joules, v_avg_fps
    except OverflowError:
        return 0, 0, 0

# --- INISIALISASI PENYIMPANAN DATA (SESSION STATE) ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- ANTARMUKA PENGGUNA (UI) ---
st.title("🎯 Raynold's Airsoft Audio Chronograph")
st.markdown("Aplikasi pengukur **Muzzle Velocity** berbasis analisis sinyal suara untuk komunitas Airsoft.")

# Sidebar untuk Input Parameter
st.sidebar.header("⚙️ Parameter Sesi")
dist = st.sidebar.number_input("Jarak ke Target (m)", min_value=1.0, value=10.0, step=0.5)
massa = st.sidebar.selectbox("Massa BB (gram)", [0.12, 0.20, 0.25, 0.28, 0.30], index=1)
suhu = st.sidebar.slider("Suhu Lokasi (°C)", 10, 40, 28)

st.sidebar.markdown("---")
st.sidebar.write("Developed by: **Raynold - FMIPA ITB**")

# Upload File Audio
uploaded_file = st.file_uploader("Unggah Rekaman Tembakan (.wav)", type=["wav"])

if uploaded_file is not None:
    with st.spinner('Menganalisis sinyal balistik...'):
        # 1. Load Audio
        y, sr = librosa.load(uploaded_file)
        
        # 2. Deteksi Onset (Lonjakan Suara)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, wait=25, pre_avg=20, post_avg=20)
        times = librosa.frames_to_time(onsets, sr=sr)
        
        if len(times) >= 2:
            # Ambil pasangan suara pertama yang terdeteksi
            t_shot = times[0]
            t_impact = times[1]
            dt_total = t_impact - t_shot
            
            # 3. Hitung Data Fisika
            v0, energy, v_avg = calculate_physics(dt_total, dist, suhu, massa)
            
            # --- TAMPILAN DISPLAY UTAMA ---
            st.markdown("### 🚀 Hasil Pengukuran Terkini")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.metric(label="MUZZLE VELOCITY (v0)", value=f"{v0:.1f} FPS")
            with c2:
                st.metric(label="ENERGY", value=f"{energy:.2f} J")
            with c3:
                st.metric(label="AVG VELOCITY", value=f"{v_avg:.1f} FPS")

            # --- VISUALISASI WAVEFORM (VERIFIKASI) ---
            st.write("#### 🔍 Verifikasi Deteksi Gelombang")
            fig, ax = plt.subplots(figsize=(12, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.5, color='blue')
            ax.vlines(t_shot, -1, 1, color='red', linestyle='--', label='Shot (Start)')
            ax.vlines(t_impact, -1, 1, color='green', linestyle='--', label='Impact (End)')
            ax.set_title(f"Delta T terdeteksi: {dt_total:.3f} detik")
            ax.legend()
            st.pyplot(fig)

            # Tombol Simpan ke Riwayat
            if st.button("➕ Simpan Hasil ke Riwayat"):
                now = datetime.now().strftime("%H:%M:%S")
                st.session_state['history'].append({
                    "Waktu": now,
                    "Jarak (m)": dist,
                    "BB (g)": massa,
                    "v0 (FPS)": round(v0, 1),
                    "Energy (J)": round(energy, 2),
                    "v_avg (FPS)": round(v_avg, 1)
                })
                st.success("Data berhasil disimpan ke tabel di bawah!")
        else:
            st.error("Gagal mendeteksi dua puncak suara. Coba gunakan target yang lebih nyaring.")

# --- BAGIAN BATCH EXPORT (HISTORY) ---
if st.session_state['history']:
    st.markdown("---")
    st.subheader("📋 Riwayat Pengujian (Batch)")
    
    # Konversi list ke DataFrame Pandas
    df_history = pd.DataFrame(st.session_state['history'])
    
    # Tampilkan Tabel
    st.dataframe(df_history, use_container_width=True)
    
    # Fitur Export ke CSV
    csv_data = df_history.to_csv(index=False).encode('utf-8')
    
    col_dl, col_clr = st.columns([1, 5])
    with col_dl:
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"chrono_export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    with col_clr:
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state['history'] = []
            st.rerun()

st.markdown("---")
st.caption("Gunakan format audio .wav dan target logam untuk akurasi maksimal.")
