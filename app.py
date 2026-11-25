import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq

# ==============================================================================
# CÁC HÀM LÕI TỪ THÀNH VIÊN 1 (GIỮ NGUYÊN THUẬT TOÁN)
# ==============================================================================
def process_uploaded_audio(uploaded_file, duration=3.0, fs_default=44100):
    if uploaded_file is not None:
        fs, audio = wavfile.read(uploaded_file)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            if info.min < 0:
                x_norm = audio.astype(np.float64) / max(abs(info.min), info.max)
            else:
                mid = (info.max + 1) / 2.0
                x_norm = (audio.astype(np.float64) - mid) / mid
        else:
            x_norm = audio.astype(np.float64)
            max_peak = np.max(np.abs(x_norm))
            if max_peak > 1.0:
                x_norm = x_norm / max_peak
        x_norm = np.clip(x_norm, -1.0, 1.0)
        source_name = uploaded_file.name
    else:
        fs = fs_default
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        x_norm = (0.6 * np.sin(2 * np.pi * 440 * t) +
                  0.3 * np.sin(2 * np.pi * 1000 * t) +
                  0.1 * np.sin(2 * np.pi * 2500 * t))
        x_norm = x_norm / np.max(np.abs(x_norm))
        source_name = "Tín hiệu mẫu mặc định (440Hz + 1000Hz + 2500Hz)"

    return fs, x_norm, source_name

def uniform_quantize_midrise(x, bits):
    levels = 2 ** bits
    delta = 2.0 / levels
    k = np.floor((x + 1.0) / delta)
    k = np.clip(k, 0, levels - 1)
    xq = -1.0 + (k + 0.5) * delta
    error = x - xq
    return xq, error, delta

def compute_metrics(original_signal, quantization_error, bits):
    ps = np.mean(original_signal ** 2)
    pq = np.mean(quantization_error ** 2)
    sqnr_exp = 10.0 * np.log10(ps / pq) if (pq > 0 and ps > 0) else np.inf
    sqnr_theo = 6.02 * bits + 1.76
    diff = abs(sqnr_exp - sqnr_theo) if np.isfinite(sqnr_exp) else np.nan
    return ps, pq, sqnr_exp, sqnr_theo, diff

def compute_fft_db(signal, fs):
    n = len(signal)
    window = np.hanning(n)
    fft_vals = rfft(signal * window)
    freqs = rfftfreq(n, d=1.0 / fs)
    magnitude = (2.0 / np.sum(window)) * np.abs(fft_vals)
    magnitude_db = 20.0 * np.log10(magnitude + 1e-12)
    return freqs, magnitude_db

def array_to_wav_bytes(signal, fs):
    """Chuyển mảng numpy [-1, 1] thành luồng bytes WAV để phát trên trình duyệt"""
    buf = io.BytesIO()
    wavfile.write(buf, fs, np.int16(np.clip(signal, -1.0, 1.0) * 32767))
    return buf.getvalue()

# ==============================================================================
# GIAO DIỆN WEB APP (PHẦN CỦA THÀNH VIÊN 2)
# ==============================================================================
st.set_page_config(page_title="Mô phỏng Lượng tử hóa Âm thanh", layout="wide")
st.title("🎛️ Mô phỏng Lượng tử hóa Tín hiệu Âm thanh (Mid-Rise)")

# 1. Thanh điều khiển bên trái (Sidebar)
with st.sidebar:
    st.header("⚙️ Cấu hình đầu vào")
    uploaded_file = st.file_uploader("Tải lên file âm thanh (.wav)", type=["wav"])
    selected_bit = st.select_slider(
        "Chọn độ phân giải Bit:",
        options=[4, 8, 12, 16],
        value=8
    )
    st.info("Nếu chưa tải file lên, hệ thống sẽ tự động dùng tín hiệu đa âm tần mẫu.")

# 2. Xử lý tín hiệu theo mức bit người dùng chọn
fs, x_orig, source_name = process_uploaded_audio(uploaded_file)
xq, err, delta = uniform_quantize_midrise(x_orig, selected_bit)
ps, pq, sqnr_exp, sqnr_theo, diff = compute_metrics(x_orig, err, selected_bit)

st.caption(f"Nguồn tín hiệu: **{source_name}** | Tần số lấy mẫu: **{fs} Hz** | Số mức lượng tử: **{2**selected_bit} mức**")

# 3. Trình phát âm thanh trực tiếp (Web Audio Player)
st.subheader("🎧 Nghe đối chiếu trực tiếp")
col_audio1, col_audio2 = st.columns(2)
with col_audio1:
    st.markdown("**Âm thanh gốc (Chuẩn hóa)**")
    st.audio(array_to_wav_bytes(x_orig, fs), format="audio/wav")
with col_audio2:
    st.markdown(f"**Âm thanh sau lượng tử hóa ({selected_bit}-bit)**")
    st.audio(array_to_wav_bytes(xq, fs), format="audio/wav")

st.divider()

# 4. Hiển thị các thông số & sai số kỹ thuật
st.subheader("📊 Thông số kỹ thuật & Sai số lượng tử")
m1, m2, m3, m4 = st.columns(4)
m1.metric("SQNR Thực nghiệm", f"{sqnr_exp:.2f} dB")
m2.metric("SQNR Lý thuyết", f"{sqnr_theo:.2f} dB", delta=f"Lệch {diff:.2f} dB", delta_color="inverse")
m3.metric("Bước lượng tử (Δ)", f"{delta:.6f}")
m4.metric("Công suất nhiễu (Pq)", f"{pq:.2e}")

st.divider()

# 5. Vẽ biểu đồ dạng sóng và phổ tần số trực tiếp lên Web
st.subheader("📈 Biểu đồ phân tích tín hiệu")
tab1, tab2, tab3 = st.tabs(["Dạng sóng & Sai số (Zoom 5ms)", "Phổ tần số FFT", "Tổng quan 4 mức Bit"])

with tab1:
    mid_idx = len(x_orig) // 2
    zoom_samples = min(len(x_orig) - mid_idx, int(0.005 * fs))
    t_zoom = np.arange(zoom_samples) / fs * 1000.0

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(t_zoom, x_orig[mid_idx:mid_idx+zoom_samples], label="Tín hiệu gốc", color="royalblue", alpha=0.8)
    ax1.step(t_zoom, xq[mid_idx:mid_idx+zoom_samples], where="mid", label=f"Lượng tử {selected_bit}-bit", color="crimson")
    ax1.set_title("So sánh dạng sóng miền thời gian")
    ax1.set_xlabel("Thời gian (ms)")
    ax1.set_ylabel("Biên độ")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(t_zoom, err[mid_idx:mid_idx+zoom_samples], color="darkgreen")
    ax2.set_title(f"Nhiễu lượng tử e(n) = x(n) - xq(n)")
    ax2.set_xlabel("Thời gian (ms)")
    ax2.set_ylabel("Sai số")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig1)

with tab2:
    fft_len = min(len(x_orig), fs * 2)
    freqs_orig, mag_orig = compute_fft_db(x_orig[:fft_len], fs)
    freqs_q, mag_q = compute_fft_db(xq[:fft_len], fs)

    fig2, ax = plt.subplots(figsize=(13, 4))
    ax.plot(freqs_orig, mag_orig, label="Phổ gốc", color="royalblue", alpha=0.7, linewidth=0.9)
    ax.plot(freqs_q, mag_q, label=f"Phổ {selected_bit}-bit", color="darkorange", alpha=0.8, linewidth=0.9)
    ax.set_title(f"Phổ tần số FFT trước và sau lượng tử hóa ({selected_bit}-bit)")
    ax.set_xlabel("Tần số (Hz)")
    ax.set_ylabel("Biên độ (dBFS)")
    ax.set_ylim([-140, 5])
    ax.set_xlim([0, fs / 2])
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig2)

with tab3:
    bits_list = [4, 8, 12, 16]
    exp_vals, theo_vals = [], []
    for b in bits_list:
        _, e_tmp, _ = uniform_quantize_midrise(x_orig, b)
        _, _, s_exp, s_theo, _ = compute_metrics(x_orig, e_tmp, b)
        exp_vals.append(s_exp)
        theo_vals.append(s_theo)

    fig3, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bits_list, theo_vals, "s--", label="SQNR Lý thuyết (6.02b + 1.76 dB)", color="navy", linewidth=2)
    ax.plot(bits_list, exp_vals, "o-", label="SQNR Thực nghiệm", color="crimson", linewidth=2)
    ax.set_title("Đường cong đối chiếu SQNR Lý thuyết vs Thực nghiệm")
    ax.set_xlabel("Độ phân giải Bit (b)")
    ax.set_ylabel("SQNR (dB)")
    ax.set_xticks(bits_list)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig3)