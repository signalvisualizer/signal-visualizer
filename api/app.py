from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.io import wavfile

app = Flask(__name__, template_folder="templates")
FS = 1500

def generate_signal(sig_type, freq, noise):
    t = np.arange(FS) / FS
    
    if sig_type == "sine":
        x = np.sin(2 * np.pi * freq * t)
    elif sig_type == "square":
        x = np.sign(np.sin(2 * np.pi * freq * t))
    elif sig_type == "triangle":
        x = 2 * np.abs(2*(t*freq - np.floor(0.5 + t*freq))) - 1
    else:
        x = np.zeros(FS)
    
    x += noise * (np.random.rand(FS)*2 - 1)
    return x


def lowpass(x):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = 0.9 * y[i-1] + 0.1 * x[i]
    return y


def highpass(x):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = x[i] - x[i-1]
    return y


def bandpass(x, lo, hi):
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), 1/FS)
    mask = (freqs >= lo) & (freqs <= hi)
    X[~mask] = 0
    return np.real(np.fft.ifft(X))


def compute_fft(x):
    X = np.fft.fft(x)
    return np.log10(np.abs(X[:FS//2]) + 1)


# -------- NEW FUNCTION (ONLY ADDITION) --------
def calculate_snr(clean, noisy):
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((noisy - clean) ** 2)

    if noise_power == 0:
        return 100.0

    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)
# ---------------------------------------------


def recommend_filter(x, fs=1500):
    X = np.abs(np.fft.fft(x))
    freqs = np.fft.fftfreq(len(x), 1/fs)

    mask = freqs >= 0
    X = X[mask]
    freqs = freqs[mask]

    total_energy = np.sum(X)

    if total_energy == 0:
        return "No dominant frequency detected"

    low_band = np.sum(X[freqs < 80])
    mid_band = np.sum(X[(freqs >= 80) & (freqs <= 250)])
    high_band = np.sum(X[freqs > 250])

    low_ratio = low_band / total_energy
    mid_ratio = mid_band / total_energy
    high_ratio = high_band / total_energy

    if low_ratio > 0.5 and high_ratio < 0.3:
        return "Low Pass Recommended"
    elif high_ratio > 0.5 and low_ratio < 0.3:
        return "High Pass Recommended"
    else:
        return "Band Pass Recommended"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "No selected file"}), 400

    import io
    sr, data = wavfile.read(io.BytesIO(f.read()))

    data = np.array(data, dtype=np.float32)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    maxv = np.max(np.abs(data)) if data.size else 0.0
    if maxv > 0:
        data = data / maxv

    data = data[:FS]

    suggestion = recommend_filter(data, fs=FS)

    filt = request.form.get('filter')
    lo = float(request.form.get('lo', 0))
    hi = float(request.form.get('hi', 0))

    y = data.copy()

    if filt == 'lp':
        y = lowpass(data)
    elif filt == 'hp':
        y = highpass(data)
    elif filt == 'bp':
        y = bandpass(data, lo, hi)

    N = len(data)

    X = np.fft.fft(data)
    fft_mag = np.log10(np.abs(X[:N//2]) + 1)

    Y = np.fft.fft(y)
    fft_mag_f = np.log10(np.abs(Y[:N//2]) + 1)

    return jsonify({
        "time_original": data.tolist(),
        "time_filtered": y.tolist(),
        "fft_original": fft_mag.tolist(),
        "fft_filtered": fft_mag_f.tolist(),
        "recommendation": suggestion
    })


@app.route("/process", methods=["POST"])
def process():
    data = request.json

    sig = data["signal"]
    freq = float(data["freq"])
    noise = float(data["noise"])
    filt = data["filter"]
    lo = float(data.get("lo", 0))
    hi = float(data.get("hi", 0))

    # clean signal (no noise)
    clean = generate_signal(sig, freq, 0)

    # noisy signal
    x = generate_signal(sig, freq, noise)

    y = x.copy()

    suggestion = recommend_filter(x, fs=FS)

    if filt == "lp":
        y = lowpass(x)
    elif filt == "hp":
        y = highpass(x)
    elif filt == "bp":
        y = bandpass(x, lo, hi)

    # SNR calculation
    snr_before = calculate_snr(clean, x)
    snr_after = calculate_snr(clean, y)

    return jsonify({
        "time_original": x.tolist(),
        "time_filtered": y.tolist(),
        "fft_original": compute_fft(x).tolist(),
        "fft_filtered": compute_fft(y).tolist(),
        "recommendation": suggestion,
        "snr_before": snr_before,
        "snr_after": snr_after
    })
@app.route("/ping")
def ping():
    return "alive"
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)