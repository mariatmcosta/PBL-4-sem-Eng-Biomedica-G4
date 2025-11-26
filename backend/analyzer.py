# analyzer.py
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend não-GUI para servidores
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend, savgol_filter

def estimate_fs_from_timestamps(ts_series):
    # ts_series: pandas Series de datetimes
    dt = ts_series.diff().dt.total_seconds().dropna()
    if len(dt) == 0:
        raise ValueError("Timestamp insuficiente para estimar frequência de amostragem.")
    mean_dt = dt.mean()
    if mean_dt == 0:
        raise ValueError("Intervalo de tempo médio é zero.")
    return 1.0 / mean_dt, mean_dt

def butter_band_bpf(signal, fs, lowcut=0.3, highcut=None, order=4):
    if highcut is None:
        highcut = fs * 0.45
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.9999)
    if low >= high:
        # filtro passa-baixa simples fallback
        b, a = butter(order, high, btype='low')
    else:
        b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

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

def analyze_csv(csv_path, output_folder):
    """
    Lê csv, executa o pipeline de pré-processamento e gera uma imagem PNG com vários subplots.
    Retorna o caminho do arquivo PNG (absoluto) e um dicionário com estatísticas.
    Espera colunas: 'Timestamp', 'Peso (kg)', 'Dorsiflexao' (nomes tolerantes).
    """
    df = pd.read_csv(csv_path, parse_dates=['Timestamp'], dayfirst=False)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # garante colunas compatíveis (tolerância a nomes)
    col_map = {}
    if 'Peso (kg)' in df.columns:
        col_map['peso'] = 'Peso (kg)'
    elif 'Peso' in df.columns:
        col_map['peso'] = 'Peso'
    else:
        raise ValueError("Coluna de peso não encontrada (esperado 'Peso (kg)' ou 'Peso')")

    if 'Dorsiflexao' in df.columns:
        col_map['ankle'] = 'Dorsiflexao'
    elif 'Dorsiflexão' in df.columns:
        col_map['ankle'] = 'Dorsiflexão'
    else:
        # tenta detectar colunas que contenham 'dorsi' insensível a maiúsculas
        candidates = [c for c in df.columns if 'dorsi' in c.lower()]
        if candidates:
            col_map['ankle'] = candidates[0]
        else:
            raise ValueError("Coluna de dorsiflexão não encontrada.")

    # extracção dos sinais
    force = df[col_map['peso']].astype(float).values
    ankle = df[col_map['ankle']].astype(float).values

    # estima fs
    fs, dt = estimate_fs_from_timestamps(df['Timestamp'])

    # remoção de NaNs se existirem
    valid_idx = (~np.isnan(force)) & (~np.isnan(ankle))
    if not np.all(valid_idx):
        force = force[valid_idx]
        ankle = ankle[valid_idx]
        ts = df['Timestamp'][valid_idx].reset_index(drop=True)
    else:
        ts = df['Timestamp']

    # detrend e suavização
    force_d = detrend(force)
    ankle_d = detrend(ankle)

    # ajuste do window para savgol (must be odd and < len)
    def safe_savgol(x, window=21, poly=3):
        if len(x) < 5:
            return x
        w = min(window, len(x)-1 if (len(x)-1)%2==1 else len(x)-2)
        if w < 3:
            return x
        if w % 2 == 0:
            w += 1
        try:
            return savgol_filter(x, w, poly)
        except:
            return x

    force_s = safe_savgol(force_d, window=21, poly=3)
    ankle_s = safe_savgol(ankle_d, window=21, poly=3)

    # filtro passa-banda
    try:
        force_f = butter_band_bpf(force_s, fs, lowcut=0.3, highcut=fs*0.45)
        ankle_f = butter_band_bpf(ankle_s, fs, lowcut=0.3, highcut=fs*0.45)
    except Exception:
        # fallback sem filtfilt
        force_f = force_s
        ankle_f = ankle_s

    # detecção de picos (heel strikes)
    try:
        min_distance = max(int(fs * 0.35), 1)
        peaks, _ = find_peaks(force_f, height=np.mean(force_f), distance=min_distance)
    except Exception:
        peaks = np.array([])

    # derivadas
    dt_sec = dt
    force_deriv = np.gradient(force_f, dt_sec)
    ankle_deriv = np.gradient(ankle_f, dt_sec)

    # FFT
    N = len(force_f)
    if N < 2:
        freqs = np.array([0.0])
        fft_force = np.array([0.0])
        fft_ankle = np.array([0.0])
    else:
        freqs = np.fft.rfftfreq(N, d=dt_sec)
        fft_force = np.abs(np.fft.rfft(force_f - np.mean(force_f)))
        fft_ankle = np.abs(np.fft.rfft(ankle_f - np.mean(ankle_f)))

    # PSD - Welch
    try:
        f_welch, Pxx = welch(force_f, fs=fs, nperseg=min(2048, N))
    except Exception:
        f_welch = np.array([0.0])
        Pxx = np.array([0.0])

    # Normalização por ciclo
    force_cycles = normalize_cycles(force_f, peaks, n_points=100)
    ankle_cycles = normalize_cycles(ankle_f, peaks, n_points=100)
    force_mean = force_cycles.mean(axis=0) if force_cycles.size else np.zeros(100)
    force_std = force_cycles.std(axis=0) if force_cycles.size else np.zeros(100)
    ankle_mean = ankle_cycles.mean(axis=0) if ankle_cycles.size else np.zeros(100)
    ankle_std = ankle_cycles.std(axis=0) if ankle_cycles.size else np.zeros(100)

    # Estatísticas
    stats = {
        "freq_estimada_hz": float(fs),
        "forca_media": float(np.mean(force_f)) if len(force_f) else None,
        "forca_max": float(np.max(force_f)) if len(force_f) else None,
        "angulo_medio": float(np.mean(ankle_f)) if len(ankle_f) else None,
        "angulo_max": float(np.max(ankle_f)) if len(ankle_f) else None,
        "n_cycles": int(force_cycles.shape[0]) if force_cycles.size else 0,
        "n_samples": int(N)
    }

    # === GERAÇÃO DO GRÁFICO COM VÁRIOS SUBPLOTS ===
    fig, axs = plt.subplots(5, 1, figsize=(14, 18), constrained_layout=True)

    # 1) Força + Heel Strikes
    axs[0].plot(ts, force_f, label='Força filtrada')
    if len(peaks):
        axs[0].plot(ts.iloc[peaks], force_f[peaks], 'r.', label='Heel-strikes')
    axs[0].set_title('Força de reação do solo (com eventos de marcha)')
    axs[0].set_ylabel('Força')
    axs[0].legend()
    axs[0].tick_params(axis='x', rotation=45)

    # 2) Derivadas
    axs[1].plot(ts, force_deriv, label='dForça/dt')
    axs[1].plot(ts, ankle_deriv, label='dÂngulo/dt')
    axs[1].set_title('Taxa de variação dos sinais')
    axs[1].set_ylabel('Derivadas')
    axs[1].legend()
    axs[1].tick_params(axis='x', rotation=45)

    # 3) FFT
    axs[2].plot(freqs, fft_force, label='FFT Força')
    axs[2].plot(freqs, fft_ankle, label='FFT Tornozelo')
    axs[2].set_xlim(0, min(15, np.max(freqs) if len(freqs) else 15))
    axs[2].set_title('Espectro de Frequência (FFT)')
    axs[2].set_xlabel('Frequência (Hz)')
    axs[2].set_ylabel('Magnitude')
    axs[2].legend()

    # 4) PSD (Welch)
    axs[3].semilogy(f_welch, Pxx)
    axs[3].set_xlim(0, min(15, np.max(f_welch) if len(f_welch) else 15))
    axs[3].set_title('Densidade Espectral de Potência (Welch)')
    axs[3].set_xlabel('Frequência (Hz)')
    axs[3].set_ylabel('Potência')

    # 5) Ciclo médio da marcha
    x = np.linspace(0, 100, len(force_mean))
    axs[4].plot(x, force_mean, label='Força média')
    axs[4].fill_between(x, force_mean-force_std, force_mean+force_std, alpha=0.3)
    axs[4].plot(x, ankle_mean, label='Ângulo médio')
    axs[4].fill_between(x, ankle_mean-ankle_std, ankle_mean+ankle_std, alpha=0.3)
    axs[4].set_title('Ciclo médio normalizado da marcha')
    axs[4].set_xlabel('% do ciclo da marcha')
    axs[4].set_ylabel('Amplitude')
    axs[4].legend()

    # salva imagem
    out_name = f"analise_marcha_{os.path.basename(csv_path).replace('.','_')}.png"
    out_path = os.path.join(output_folder, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return out_path, stats