import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq


class ExtractFeatures:
    def __init__(
            self, 
            windows=[5, 10, 30], #długości okien czasowych
            network_cols=["pkt_count", "total_bytes", "avg_pkt_len"], #cechy ruchu sieciowego
            process_cols=["LIT101.Pv", "FIT101.Pv"], # czujniki procesowe
            binary_state_cols=None, # stany binarne 
            fft_cols=None, # sygnały cigąłe z cech widmowych FFT
            fft_windows=30, # długość okna próbek użycia do liczenia FFT
            lag_steps=[1, 5, 10] # lista przesunięć czasowych dla cech typu lat
            ):
        self.windows = windows
        self.network_cols = network_cols
        self.process_cols = process_cols
        self.binary_state_cols = binary_state_cols or []
        self.fft_cols = fft_cols or []
        self.fft_windows = fft_windows
        self.lag_steps = lag_steps



    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_features = df.copy()
 
        if df_features[self.network_cols].isna().any().any():
            n_missing = df_features[self.network_cols[0]].isna().sum()
            raise ValueError(
                f"Kolumny sieciowe zawierają {n_missing} braków danych (NaN). "
                "Usuń wiersze bez dopasowania sieciowego przed ekstrakcją cech, "
                "np. df = df.dropna(subset=network_cols)."
            )
 
        df_features = self._basic_network_features(df_features)
        df_features = self._rolling_statistics(df_features)
        df_features = self._trend_features(df_features)
        df_features = self._distribution_features(df_features)
        df_features = self._correlation_features(df_features)
        df_features = self._lag_features(df_features)
 
        if self.binary_state_cols:
            df_features = self._state_change_features(df_features)
 
        if self.fft_cols:
            df_features = self._fft_features(df_features)
 
        return df_features


    @staticmethod
    def _attach(df, new_cols: dict) -> pd.DataFrame:
        """Dokleja słownik nowych kolumn do df jednym pd.concat (bez fragmentacji)."""
        new_df = pd.DataFrame(new_cols, index=df.index)
        new_df = new_df.replace([np.inf, -np.inf], 0)
        return pd.concat([df, new_df], axis=1)

    
    # Podstawowe cechy ruchu sieciowego (bez okna czasowego)
    def _basic_network_features(self, df):
        new_cols = {}
        new_cols["bytes_per_pkt"] = df["total_bytes"] / (df["pkt_count"] + 1e-5)
        new_cols["pkt_count_diff"] = df["pkt_count"].diff().fillna(0)
        new_cols["pkt_count_pct_change"] = (
            df["pkt_count"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        )
        return self._attach(df, new_cols)

    # Cechy statystyczne w oknie
    def _rolling_statistics(self, df):
        target_cols = list(dict.fromkeys(self.network_cols + self.process_cols))
        new_cols = {}
 
        for w in self.windows:
            for col in target_cols:
                roll = df[col].rolling(window=w, min_periods=1)
                col_min = roll.min()
                col_max = roll.max()
                new_cols[f"{col}_mean_{w}s"] = roll.mean()
                new_cols[f"{col}_std_{w}s"] = roll.std().fillna(0)
                new_cols[f"{col}_min_{w}s"] = col_min
                new_cols[f"{col}_max_{w}s"] = col_max
                new_cols[f"{col}_range_{w}s"] = col_max - col_min
 
        return self._attach(df, new_cols)

    # Cechy trendu
    def _trend_features(self, df):
        target_cols = list(dict.fromkeys(self.network_cols + self.process_cols))
        new_cols = {}
 
        for w in self.windows:
            for col in target_cols:
                new_cols[f"{col}_ewm_{w}s"] = df[col].ewm(span=w, adjust=False).mean()
                # rate of change: różnica między wartością bieżącą a sprzed w kroków
                new_cols[f"{col}_roc_{w}s"] = df[col].diff(periods=w).fillna(0)
 
        return self._attach(df, new_cols)

    # Cechy rozkładu: skłonność _skew), kurtoza, w oknie
    def _distribution_features(self, df):
        w = max(self.windows)
        target_cols = list(dict.fromkeys(self.network_cols + self.process_cols))
        new_cols = {}
 
        for col in target_cols:
            roll = df[col].rolling(window=w, min_periods=3)
            new_cols[f"{col}_skew_{w}s"] = roll.skew().fillna(0)
            new_cols[f"{col}_kurt_{w}s"] = roll.kurt().fillna(0)
 
        return self._attach(df, new_cols)

    # Cechy korelacyjne 
    def _correlation_features(self, df):
        new_cols = {}
        for w in self.windows:
            for net_col in self.network_cols:
                for proc_col in self.process_cols:
                    new_cols[f"corr_{net_col}_{proc_col}_{w}s"] = (
                        df[net_col]
                        .rolling(window=w, min_periods=2)
                        .corr(df[proc_col])
                        .fillna(0)
                    )
        return self._attach(df, new_cols)

    # Cechy opóźnień czasowych 
    def _lag_features(self, df):
        target_cols = list(dict.fromkeys(self.network_cols + self.process_cols))
        new_cols = {}
 
        for lag in self.lag_steps:
            for col in target_cols:
                new_cols[f"{col}_lag_{lag}"] = df[col].shift(lag).bfill()
 
        return self._attach(df, new_cols)

    # Cechy stanowe
    def _state_change_features(self, df):
        new_cols = {}
        for col in self.binary_state_cols:
            if col not in df.columns:
                continue
            changes = df[col].diff().fillna(0).abs()
            for w in self.windows:
                new_cols[f"{col}_switch_count_{w}s"] = (
                    changes.rolling(window=w, min_periods=1).sum()
                )
        return self._attach(df, new_cols)

    # Cechy częstotliwościowe FFT 
    # nawiązuje do analizy STFT (rozdz 2) jako cechy tabelaryczne do ML
    # dla każdego okna liczy dominującą częstotliwość i energie w pasmach niskich i wysokich częstotliwości
    def _fft_features(self, df, fs=1.0):
        n = self.fft_window
        new_cols = {}
 
        for col in self.fft_cols:
            if col not in df.columns:
                continue
 
            peak_freqs = np.zeros(len(df))
            low_energy = np.zeros(len(df))
            high_energy = np.zeros(len(df))
 
            values = df[col].to_numpy()
 
            for i in range(len(df)):
                start = max(0, i - n + 1)
                window_vals = values[start : i + 1]
 
                if len(window_vals) < 4:
                    continue
 
                window_vals = window_vals - window_vals.mean()
                spectrum = np.abs(rfft(window_vals))
                freqs = rfftfreq(len(window_vals), d=1.0 / fs)
 
                if len(spectrum) <= 1:
                    continue
 
                # pomijamy składową stałą (DC, indeks 0) przy szukaniu piku
                peak_idx = np.argmax(spectrum[1:]) + 1
                peak_freqs[i] = freqs[peak_idx]
 
                mid = len(spectrum) // 2
                low_energy[i] = spectrum[1:mid].sum() if mid > 1 else 0
                high_energy[i] = spectrum[mid:].sum()
 
            new_cols[f"{col}_fft_peak_freq"] = peak_freqs
            new_cols[f"{col}_fft_low_energy"] = low_energy
            new_cols[f"{col}_fft_high_energy"] = high_energy
    
        return self._attach(df, new_cols)

    



