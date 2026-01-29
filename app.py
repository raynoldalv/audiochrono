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
    Menghitung Muzzle Velocity (v0) dengan model hambatan udara eksponensial.
    """
    rho = 1.225      # Massa jenis udara (kg/m3)
    Cd = 0.47        # Koefisien drag bola (Sphere)
    radius = 0.003   # Radius BB 6mm (m)
    area = math.pi * (radius**2)
    m = mass_gram / 1000  # Konversi ke kg
    
    # 1. Kecepatan Suara (v_s) berdasarkan suhu
    v_s = 331.4 * math.sqrt(1 + t_env / 273.15)
    
    # 2. Waktu rambat suara kembali
    t_sound_return = s / v_s
    t_flight = dt_total - t_sound_return
    
    if t_flight <= 0:
        return 0, 0, 0
    
    # 3. Konstanta Drag (k)
    k = 0.5 * rho * Cd * area
    
    try:
        # Rumus v0: v0 = (m * (exp(ks/m) - 1)) / (k * t_flight)
        v0_m_s = (m * (math.exp(k * s / m) - 1)) / (k * t_flight)
        v0_fps = v0_m_s * 3.28084
        
        # 4. Energi Kinetik (Joule)
        joules = 0.5 * m * (v0_m_s**2)
        
        # 5. Kecepatan Rata-rata (v_avg)
        v_avg_fps = (s / t_flight) * 3.28084
        
        return v0_fps, joules, v_avg_fps
    except:
        return 0, 0, 0

# --- INISIALISASI RIWAYAT (SESSION STATE) ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- ANTARMUKA PENGGUNA (UI) ---
st.title("🎯 Raynold's Airsoft Audio Chronograph")
st.markdown("Aplikasi analisis balistik berbasis suara untuk komunitas Airsoft.")

# Sidebar Parameter
st.sidebar.header("⚙️ Konfigurasi")
dist = st.sidebar.number_input("Jarak ke Target (m)", min_value=1.0, value=10.0, step=0.5)
massa = st.sidebar.selectbox("Massa BB (gram)", [0.12, 0.20, 0.25, 0.28, 0.30], index=1)
suhu = st.sidebar.slider("Suhu Lokasi (°C)", 10, 40, 28)

st.sidebar.markdown("---")
st.sidebar.write("Project by: **Raynold - FMIPA ITB**")

# --- FILE UPLOADER (MOBILE FRIENDLY) ---
# Menghapus 'type' agar browser mobile membuka menu 'Files/Documents'
uploaded_file = st.file_uploader("Unggah Rekaman Tembakan (Gunakan menu 'Files' di HP untuk memilih rekaman audio.)")

if uploaded_file is not None:
    # Cek ekstensi file secara manual
    fname = uploaded_file.name.lower()
    if fname.endswith(('.wav', '.mp3', '.m4a', '.aac', '.ogg')):
        with st.spinner('Menganalisis sinyal...'):
            # Load audio (Librosa otomatis handle format via audioread)
            y, sr = librosa.load(uploaded_file)
            
            # Deteksi Puncak Suara
            onsets = librosa.onset.onset_detect(y=y, sr=sr, wait=25)
            times = librosa.frames_to_time(onsets, sr=sr)
            
            if len(times) >= 2:
                dt_total = times[1] - times[0]
                v0, energy, v_avg = calculate_physics(dt_total, dist, suhu, massa)
                
                # Menampilkan Hasil
                st.subheader("🚀 Hasil Pengukuran")
                c1, c2, c3 = st.columns(3)
                c1.metric("MUZZLE VELOCITY (v0)", f"{v0:.1f} FPS")
                c2.metric("ENERGY (J)", f"{energy:.2f} J")
                c3.metric("AVG VELOCITY", f"{v_avg:.1f} FPS")

                # Grafik Verifikasi
                st.write("#### 🔍 Verifikasi Waveform")
                fig, ax = plt.subplots(figsize=(10, 3))
                librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.5)
                ax.vlines([times[0], times[1]], -1, 1, color=['red', 'green'], linestyle='--', label=['Shot', 'Impact'])
                ax.legend()
                st.pyplot(fig)

                # Tombol Simpan
                if st.button("➕ Simpan ke Riwayat"):
                    st.session_state['history'].append({
                        "Waktu": datetime.now().strftime("%H:%M:%S"),
                        "BB (g)": massa,
                        "v0 (FPS)": round(v0, 1),
                        "Joule": round(energy, 2),
                        "Jarak (m)": dist
                    })
                    st.rerun()
            else:
                st.error("Gagal mendeteksi puncak suara. Gunakan target logam agar suara lebih tajam.")
    else:
        st.error("Format file tidak didukung. Mohon gunakan .wav, .mp3, atau .m4a.")

# --- BAGIAN BATCH EXPORT ---
if st.session_state['history']:
    st.markdown("---")
    st.subheader("📋 Riwayat Pengujian (Batch)")
    df = pd.DataFrame(st.session_state['history'])
    st.dataframe(df, use_container_width=True)
    
    # Download Button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data CSV",
        data=csv,
        file_name=f"chrono_raynold_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
    
    if st.button("🗑️ Hapus Riwayat"):
        st.session_state['history'] = []
        st.rerun()
