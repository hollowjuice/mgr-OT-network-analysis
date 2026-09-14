from src import MergeData, ExtractFeatures

swat_path = 'data\\SWaT_processed_data.csv'
pcap_path = 'data\\pcap_network_features.csv'

merge = MergeData(swat_path, pcap_path)
df_merged = merge.merge()
print("df_merged shape:", df_merged.shape)

ef = ExtractFeatures(
    windows=[5, 10, 30],
    network_cols=["pkt_count", "total_bytes", "avg_pkt_len"],
    process_cols=["LIT101.Pv", "FIT101.Pv"],       # dobierz kluczowe czujniki z Historian
    binary_state_cols=["P101.Status", "MV101.Status"],  # zawory/pompy z Twojego zbioru
    fft_cols=["FIT101.Pv"],                          # nawiązanie do STFT z rozdz. 2
    fft_windows=30,
    lag_steps=[1, 5, 10],
)
df_out = ef.extract_features(df_merged)

#print(df_out.head())
#print(df_features.describe())
#print(df_out.select_dtypes(include='number').describe().T) 
df_out.to_csv('data/features_extracted.csv', index=False)
