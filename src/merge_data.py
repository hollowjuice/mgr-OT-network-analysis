import pandas as pd 

class MergeData:
    def __init__(self, csv_path: str, pcap_path: str):
        self.csv_path = csv_path
        self.pcap_path = pcap_path

    def merge(self) -> pd.DataFrame:
        # 1. Wczytanie danych z plików
        df_csv = pd.read_csv(self.csv_path)
        df_pcap = pd.read_csv(self.pcap_path)

        # 2. Ujednolicenie czasu SWaT (przeliczenie z GMT+8 na UTC i narzucenie typu datetime64[ns])
        df_csv["datetime"] = (
            pd.to_datetime(df_csv["Timestamp_GMT8"])
            .dt.tz_localize("Etc/GMT-8")  # Określenie strefy GMT+8
            .dt.tz_convert("UTC")  # Konwersja do UTC
            .dt.tz_localize(None)  # Usunięcie znacznika strefy (naive datetime)
            .astype("datetime64[ns]")  # Naprawa MergeError: wymuszenie typu [ns]
        )

        # 3. Ujednolicenie czasu PCAP (konwersja sekundowego timestampu Unix na UTC)
        df_pcap["datetime"] = (
            pd.to_datetime(df_pcap["sec_timestamp"], unit="s")
            .astype("datetime64[ns]")  # Naprawa MergeError: wymuszenie typu [ns]
        )

        # 4. Sortowanie po czasie (wymóg konieczny dla merge_asof)
        df_csv = df_csv.sort_values(by="datetime")
        df_pcap = df_pcap.sort_values(by="datetime")

        # 5. Połączenie danych po czasie
        df_merged = pd.merge_asof(
            df_csv,
            df_pcap,
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta("2s"),  # Tolerancja dopasowania rekordów (np. 5 sekund)
        )

        return df_merged