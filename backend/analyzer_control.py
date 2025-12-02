# backend/analyzer_control.py
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem GUI para servidores
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend, savgol_filter

def _safe_savgol(signal, window_length=21, polyorder=3):
    """Aplica savgol de forma segura: janela ímpar, <= len-1 e >=3. Se não aplicável, retorna sinal original."""
    n = len(signal)
    if n < 5:
        return signal
    # tentar adaptar janela
    wl = int(window_length)
    if wl >= n:
        wl = n - 1
    # garantir ímpar
    if wl % 2 == 0:
        wl -= 1
    if wl < 3:
        wl = 3
    if wl >= n:
        # último recurso
        if n % 2 == 1:
            wl = n - 2
        else:
            wl = n - 3
        if wl < 3:
            return signal
    try:
        return savgol_filter(signal, wl, polyorder)
    except Exception:
        return signal

def _butter_band_filter(signal, fs, lowcut=0.3, highcut=None, order=4):
    """Aplica filtro Butterworth passa-banda com segurança de Nyquist; se low >= high, faz passa-baixa."""
    if highcut is None:
        highcut = fs * 0.45
    nyq = 0.5 * fs
    # evitar division by zero
    if nyq <= 0:
        return signal
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.9999)
    try:
        if low >= high:
            # fallback: passa-baixa em high
            b, a = butter(order, high, btype='low')
        else:
            b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)
    except Exception:
        return signal

def _normalize_cycles(signal, peaks, n_points=100):
    cycles = []
    for i in range(len(peaks)-1):
        start = peaks[i]
        end = peaks[i+1]
        if end - start < 3:
            continue
        seg = signal[start:end]
        x_old = np.linspace(0, 100, num=len(seg))
        x_new = np.linspace(0, 100, num=n_points)
        cycles.append(np.interp(x_new, x_old, seg))
    if len(cycles) == 0:
        return np.empty((0, n_points))
    return np.array(cycles)

def analyze_csv_control(csv_path, output_folder):
    """
    Processa CSV do grupo controle e salva PNG em output_folder.
    Retorna (out_path, stats_dict).
    Espera coluna 'Timestamp' (datetime), 'Peso (kg)' (ou 'Peso') e 'Dorsiflexao' (ou similar contendo 'dorsi').
    """
    # leitura
    df = pd.read_csv(csv_path, parse_dates=['Timestamp'], dayfirst=False)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # mapear colunas com tolerância a nomes
    # peso
    if 'Peso (kg)' in df.columns:
        peso_col = 'Peso (kg)'
    elif 'Peso' in df.columns:
        peso_col = 'Peso'
    else:
        raise ValueError("Coluna de peso não encontrada (esperado 'Peso (kg)' ou 'Peso').")

    # dorsiflexão / tornozelo
    if 'Dorsiflexao' in df.columns:
        ankle_col = 'Dorsiflexao'
    elif 'Dorsiflexão' in df.columns:
        ankle_col = 'Dorsiflexão'
    else:
        candidates = [c for c in df.columns if 'dorsi' in c.lower() or 'tornoz' in c.lower() or 'ankle' in c.lower()]
        if candidates:
            ankle_col = candidates[0]
        else:
            # último recurso: pega segunda coluna numérica além do peso (não ideal)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != peso_col]
            if numeric_cols:
                ankle_col = numeric_cols[0]
            else:
                raise ValueError("Coluna de dorsiflexão não encontrada.")

    # extrair sinais
    force = df[peso_col].astype(float).values
    ankle = df[ankle_col].astype(float).values

    # estimar dt e fs de forma robusta
    dt_series = df['Timestamp'].diff().dt.total_seconds().dropna()
    if len(dt_series) == 0:
        raise ValueError("Timestamp insuficiente para estimar frequência de amostragem.")
    dt = dt_series.mean()
    if dt == 0 or np.isnan(dt) or not np.isfinite(dt):
        raise ValueError("Delta de tempo inválido (dt é zero / NaN).")
    fs = 1.0 / dt

    # detrend
    force_d = detrend(force)
    ankle_d = detrend(ankle)

    # suavização segura
    force_s = _safe_savgol(force_d, window_length=21, polyorder=3)
    ankle_s = _safe_savgol(ankle_d, window_length=21, polyorder=3)

    # filtro passa-banda (0.3 - fs*0.45)
    force_f = _butter_band_filter(force_s, fs, lowcut=0.3, highcut=fs*0.45, order=4)
    ankle_f = _butter_band_filter(ankle_s, fs, lowcut=0.3, highcut=fs*0.45, order=4)

    # detectar picos (heel-strikes)
    try:
        min_distance = max(int(fs * 0.4), 1)
        height_thr = np.mean(force_f) if np.isfinite(np.mean(force_f)) else None
        height_arg = {'height': height_thr} if height_thr is not None else {}
        peaks, _ = find_peaks(force_f, distance=min_distance, **height_arg)
    except Exception:
        peaks = np.array([])

    # derivadas
    force_deriv = np.gradient(force_f, dt)
    ankle_deriv = np.gradient(ankle_f, dt)

    # FFT
    N = len(force_f)
    if N < 2:
        freqs = np.array([0.0])
        fft_force = np.array([0.0])
        fft_ankle = np.array([0.0])
    else:
        freqs = np.fft.rfftfreq(N, d=dt)
        fft_force = np.abs(np.fft.rfft(force_f - np.mean(force_f)))
        fft_ankle = np.abs(np.fft.rfft(ankle_f - np.mean(ankle_f)))

    # PSD - Welch (tratamento seguro de nperseg)
    try:
        nperseg = min(2048, N)
        if nperseg < 8:
            f_welch, Pxx = np.array([]), np.array([])
        else:
            f_welch, Pxx = welch(force_f, fs=fs, nperseg=nperseg)
    except Exception:
        f_welch, Pxx = np.array([]), np.array([])

    # normalização por ciclo
    force_cycles = _normalize_cycles(force_f, peaks, n_points=100)
    ankle_cycles = _normalize_cycles(ankle_f, peaks, n_points=100)

    if force_cycles.size:
        force_mean = np.mean(force_cycles, axis=0)
        force_std = np.std(force_cycles, axis=0)
    else:
        force_mean = np.zeros(100)
        force_std = np.zeros(100)

    if ankle_cycles.size:
        ankle_mean = np.mean(ankle_cycles, axis=0)
        ankle_std = np.std(ankle_cycles, axis=0)
    else:
        ankle_mean = np.zeros(100)
        ankle_std = np.zeros(100)

    # reconstrução do sinal periódico
    num_passadas = max(len(peaks) - 1, 1)
    reconstructed_force = np.tile(force_mean, num_passadas)
    reconstructed_ankle = np.tile(ankle_mean, num_passadas)

    # duração média do ciclo
    try:
        if len(peaks) >= 2:
            cycle_duration = (df['Timestamp'].iloc[peaks[1]-1] - df['Timestamp'].iloc[peaks[0]]).total_seconds()
        else:
            cycle_duration = 1.0
    except Exception:
        cycle_duration = 1.0

    t_recon = np.linspace(0, num_passadas * cycle_duration, len(reconstructed_force)) if len(reconstructed_force) else np.array([])

    # estatísticas
    stats = {
        "freq_estimada_hz": float(fs),
        "forca_media": float(np.mean(force_f)) if len(force_f) else None,
        "forca_max": float(np.max(force_f)) if len(force_f) else None,
        "angulo_medio": float(np.mean(ankle_f)) if len(ankle_f) else None,
        "angulo_max": float(np.max(ankle_f)) if len(ankle_f) else None,
        "n_peaks": int(len(peaks)),
        "n_cycles": int(force_cycles.shape[0]) if force_cycles.size else 0,
        "n_samples": int(N)
    }

    # --- plots (igual ao seu script original) ---
    fig, axs = plt.subplots(5, 1, figsize=(14, 20), sharex=False)

    # 1) Força Bruta (do CSV)
    axs[0].plot(df['Timestamp'], df[peso_col].values, label='Força Bruta')
    axs[0].set_title('1. Força de Reação do Solo (Dados Brutos do CSV)')
    axs[0].set_ylabel('Peso (kg)')
    axs[0].legend()
    axs[0].tick_params(axis='x', rotation=45)

    # 2) Ângulo do tornozelo (filtrado)
    axs[1].plot(df['Timestamp'], ankle_f, 'g-', label='Dorsiflexão filtrada')
    axs[1].set_title('2. Ângulo do Tornozelo (Dorsiflexão) ao longo do Tempo')
    axs[1].set_ylabel('Ângulo (unidade normalizada)')
    axs[1].legend()
    axs[1].tick_params(axis='x', rotation=45)

    # 3) FFT
    axs[2].plot(freqs, fft_force, label='FFT Força')
    axs[2].plot(freqs, fft_ankle, label='FFT Tornozelo')
    axs[2].set_xlim(0, min(15, np.max(freqs) if len(freqs) else 15))
    axs[2].set_title('3. Espectro de Frequência (FFT)')
    axs[2].set_xlabel('Frequência (Hz)')
    axs[2].set_ylabel('Magnitude')
    axs[2].legend()

    # 4) Ciclo médio da marcha (normalizado 0-100%)
    x = np.linspace(0, 100, len(force_mean))
    axs[3].plot(x, force_mean, label='Força média')
    axs[3].fill_between(x, force_mean-force_std, force_mean+force_std, alpha=0.3, label='Desvio Padrão Força')
    axs[3].plot(x, ankle_mean, label='Ângulo médio')
    axs[3].fill_between(x, ankle_mean-ankle_std, ankle_mean+ankle_std, alpha=0.3, label='Desvio Padrão Ângulo')
    axs[3].set_title('4. Ciclo Médio Normalizado da Marcha (0-100% do Passo)')
    axs[3].set_xlabel('% do ciclo da marcha')
    axs[3].set_ylabel('Amplitude Normalizada')
    axs[3].legend()

    # 5) Sinal reconstruído
    if t_recon.size:
        axs[4].plot(t_recon, reconstructed_force, label='Força reconstruída')
        axs[4].plot(t_recon, reconstructed_ankle, label='Ângulo reconstruído')
    else:
        axs[4].plot([], [], label='Sem sinal reconstruído')
    axs[4].set_title('5. Sinal Reconstruído da Passada (Períodos Repetidos do Ciclo Médio)')
    axs[4].set_xlabel('Tempo Artificial (s)')
    axs[4].set_ylabel('Amplitude')
    axs[4].legend()

    plt.tight_layout()

    # salvar
    base = os.path.basename(csv_path)
    name_no_ext = os.path.splitext(base)[0]
    out_name = f"analise_control_{name_no_ext}.png"
    out_path = os.path.join(output_folder, out_name)
    # garantir pasta
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return out_path, stats