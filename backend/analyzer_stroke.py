import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend, savgol_filter


def analyze_csv_stroke(csv_path, output_folder):
    df = pd.read_csv(csv_path, parse_dates=['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # outliers (peso)
    if 'Peso (kg)' not in df.columns:
        raise ValueError("Coluna 'Peso (kg)' não encontrada no CSV")

    Q1 = df['Peso (kg)'].quantile(0.25)
    Q3 = df['Peso (kg)'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df.loc[(df['Peso (kg)'] < lower) | (df['Peso (kg)'] > upper), 'Peso (kg)'] = np.nan
    df['Peso (kg)'] = df['Peso (kg)'].interpolate().fillna(method='bfill').fillna(method='ffill')

    # fs
    dt_series = df['Timestamp'].diff().dt.total_seconds()
    dt = dt_series.mean()
    if np.isnan(dt) or dt == 0:
        dt = 1.0
    fs = 1.0 / dt

    force = df['Peso (kg)'].astype(float).values
    ankle_col = 'Dorsiflexao' if 'Dorsiflexao' in df.columns else ('Dorsiflexão' if 'Dorsiflexão' in df.columns else [c for c in df.columns if 'dorsi' in c.lower()][0])
    ankle = df[ankle_col].astype(float).values

    # clean NaN/Inf
    if np.isnan(force).any() or np.isinf(force).any():
        force = np.nan_to_num(force)
    if np.isnan(ankle).any() or np.isinf(ankle).any():
        ankle = np.nan_to_num(ankle)

    force_detrended = detrend(force)
    ankle_detrended = detrend(ankle)

    def safe_savgol(signal, window_length=21, polyorder=3):
        n = len(signal)
        if n < 5:
            return signal
        wl = min(window_length, n if n % 2 == 1 else n - 1)
        if wl < 3:
            wl = 3
        if wl % 2 == 0:
            wl -= 1
            if wl < 3:
                wl = 3
        if wl >= n:
            wl = n - 2 if (n - 2) % 2 == 1 else n - 3
            if wl < 3:
                wl = 3
        try:
            return savgol_filter(signal, wl, polyorder)
        except Exception:
            return signal

    force_smoothed = safe_savgol(force_detrended)
    ankle_smoothed = safe_savgol(ankle_detrended)

    cutoff = min(20.0, 0.45 * fs)
    if cutoff <= 0:
        cutoff = fs * 0.45
    b, a = butter(4, cutoff, btype='low', fs=fs)

    force_f = filtfilt(b, a, force_smoothed)
    ankle_f = filtfilt(b, a, ankle_smoothed)
    force_f[force_f < 0] = 0

    min_peak_distance = max(1, int(fs * 0.4))
    height_thr = np.mean(force_f) if np.isfinite(np.mean(force_f)) else None
    height_arg = {'height': height_thr} if height_thr is not None else {}
    peaks, props = find_peaks(force_f, distance=min_peak_distance, **height_arg)

    if len(peaks) < 2:
        # fall back: try without height
        peaks, props = find_peaks(force_f, distance=min_peak_distance)

    force_deriv = np.gradient(force_f, dt)
    ankle_deriv = np.gradient(ankle_f, dt)

    N = len(force_f)
    freqs = np.fft.rfftfreq(N, d=dt)
    fft_force = np.abs(np.fft.rfft(force_f - np.mean(force_f)))
    fft_ankle = np.abs(np.fft.rfft(ankle_f - np.mean(ankle_f)))

    nperseg = min(2048, N)
    if nperseg < 8:
        f_welch, Pxx = np.array([]), np.array([])
    else:
        f_welch, Pxx = welch(force_f, fs=fs, nperseg=nperseg)

    def normalize_cycles(signal, peaks, n_points=100):
        cycles = []
        for i in range(len(peaks)-1):
            start = peaks[i]
            end = peaks[i+1]
            if end - start < 3:
                continue
            seg = signal[start:end]
            x_old = np.linspace(0, 100, len(seg))
            x_new = np.linspace(0, 100, n_points)
            cycles.append(np.interp(x_new, x_old, seg))
        if len(cycles) == 0:
            return np.empty((0, n_points))
        return np.array(cycles)

    force_cycles = normalize_cycles(force_f, peaks)
    ankle_cycles = normalize_cycles(ankle_f, peaks)

    force_mean = force_cycles.mean(axis=0) if force_cycles.size else np.zeros(100)
    force_std = force_cycles.std(axis=0) if force_cycles.size else np.zeros(100)
    ankle_mean = ankle_cycles.mean(axis=0) if ankle_cycles.size else np.zeros(100)
    ankle_std = ankle_cycles.std(axis=0) if ankle_cycles.size else np.zeros(100)

    num_passadas = max(len(peaks) - 1, 1)
    reconstructed_force = np.tile(force_mean, num_passadas)
    reconstructed_ankle = np.tile(ankle_mean, num_passadas)

    try:
        cycle_duration = (df['Timestamp'].iloc[peaks[1]-1] - df['Timestamp'].iloc[peaks[0]]).total_seconds()
    except Exception:
        cycle_duration = (df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]).total_seconds() / max(1, num_passadas)

    t_recon = np.linspace(0, num_passadas * cycle_duration, len(reconstructed_force))

    stats = {
        "freq_estimada_hz": float(fs),
        "forca_media": float(np.mean(force_f)) if len(force_f) else None,
        "forca_max": float(np.max(force_f)) if len(force_f) else None,
        "angulo_medio": float(np.mean(ankle_f)) if len(ankle_f) else None,
        "angulo_max": float(np.max(ankle_f)) if len(ankle_f) else None,
        "n_cycles": int(force_cycles.shape[0]) if force_cycles.size else 0,
        "n_samples": int(N)
    }

    fig, axs = plt.subplots(5, 1, figsize=(14, 20))

    axs[0].plot(df['Timestamp'], force_f, label='Força filtrada e sem negativos')
    axs[0].set_title('1. Força de Reação do Solo (Outliers removidos)')

    axs[1].plot(df['Timestamp'], ankle_f, 'g-', label='Dorsiflexão filtrada')
    axs[1].set_title('2. Ângulo do Tornozelo')

    axs[2].plot(freqs, fft_force, label='FFT Força')
    axs[2].plot(freqs, fft_ankle, label='FFT Tornozelo')
    axs[2].set_xlim(0, min(15, fs/2))
    axs[2].set_title('3. FFT')

    x = np.linspace(0, 100, len(force_mean))
    axs[3].plot(x, force_mean, label='Força média')
    axs[3].fill_between(x, force_mean-force_std, force_mean+force_std, alpha=0.3)
    axs[3].plot(x, ankle_mean, label='Ângulo médio')
    axs[3].fill_between(x, ankle_mean-ankle_std, ankle_mean+ankle_std, alpha=0.3)
    axs[3].set_title('4. Ciclo médio da marcha')

    axs[4].plot(t_recon, reconstructed_force, label='Força reconstruída')
    axs[4].plot(t_recon, reconstructed_ankle, label='Ângulo reconstruído')
    axs[4].set_title('5. Sinal reconstruído')

    plt.tight_layout()

    out_name = f"analise_stroke_{os.path.basename(csv_path).replace('.','_')}.png"
    out_path = os.path.join(output_folder, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return out_path, stats