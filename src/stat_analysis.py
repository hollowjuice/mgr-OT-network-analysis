import numpy as np 
import pandas as pd
from scipy import stats

try:
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

class StatisticalAnalyzer:
    ''' 
    moduł analizy statystycznej do rozdziału 3.2

    obejmuje:
    - statysyki opisowe
    - test kołmogorowa-smirnowa
    - test adf
    - funkcje autokorelacji i częściowej autokorelacji
    - macierz korealcji cech oraz identyfikacja par silnie skorelowanych
    '''

    def __init__(self, target_col="y"):
        self.target_col = target_col

    # domyślny wybór kolumn numerycznych
    def _default_feature_cols(self, df):
        return (
            df.select_dtypes(include=[np.number])
            .columns.drop(self.target_col, errors="ignore")
            .tolist()
        )

    # statystyki opisowe
    def descriptive_statistics(self, df: pd.DataFrame, feature_cols=None) -> dict:
        feature_cols = feature_cols or self._default_feature_cols(df)
        results = {}

        for cls in sorted(df[self.target_col].unique()):
            subset = df.loc[df[self.target_col] == cls, feature_cols]
            results[cls] = subset.describe().T[["mean", "std", "min", "50%", "max"]]

        return results

    # test kołmogorowa-smirnowa
    def ks_test(self, df: pd.DataFrame, feature_cols=None, alpha=0.05) -> pd.DataFrame:
        feature_cols = feature_cols or self._default_feature_cols(df)
        normal = df.loc[df[self.target_col] == 0]
        anomaly = df.loc[df[self.target_col] == 1]

        rows = []
        for col in feature_cols:
            stat, p_value = stats.ks_2samp(normal[col], anomaly[col])
            rows.append(
                {
                    "feature":col,
                    "ks_statistic": stat,
                    "p_value": p_value,
                    "significant": p_value < alpha,
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values(by="ks_statistic", ascending=False)
            .reset_index(drop=True)
        )

    # test adf
    def adf_test(self, series: pd.Series, alpha=0.05) -> dict:
        if not _STATSMODELS_AVAILABLE:
            raise ImportError(
                "adf_test() wymaga pakietu statsmodels. Zainstaluj: "
                "pip install statsmodels"
            )
 
        series_clean = series.dropna()
        result = adfuller(series_clean, autolag="AIC", result_object=False)
        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "n_lags": result[2],
            "n_obs": result[3],
            "critical_values": result[4],
            "is_stationary": result[1] < alpha,
        }

    def adf_test_batch(self, df: pd.DataFrame, feature_cols=None, alpha=0.05) -> pd.DataFrame:
        feature_cols = feature_cols or self._default_feature_cols(df)
        rows = []
        for col in feature_cols:
            try:
                res = self.adf_test(df[col], alpha=alpha)
                rows.append({"feature": col, "adf_statistic": res["adf_statistic"],
                             "p_value": res["p_value"], "is_stationary": res["is_stationary"]})
            except Exception as e:
                rows.append({"feature": col, "adf_statistic": np.nan,
                             "p_value": np.nan, "is_stationary": None})

        return pd.DataFrame(rows)

    # acf / pcaf autokorelacja 
    def acf_pacf(self, series: pd.Series, nlags=40) -> dict:
        if not _STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels error"
            )

        series_clean = series.dropna()
        acf_values = acf(series_clean, nlags=nlags, fft=True)
        pacf_vals = pacf(series_clean, nlags=nlags)
        return{"acf": acf_values, "pacf": pacf_vals}

    # macierz korelacji cech, identyfikacja redundancji
    def correlation_matrix(self, df: pd.DataFrame, feature_cols=None, method="pearson"):
        """Macierz korelacji cech - wejście np. do heatmapy w rozdziale 3.2."""
        feature_cols = feature_cols or self._default_feature_cols(df)
        return df[feature_cols].corr(method=method)

    def top_correlated_pairs(self, corr_matrix: pd.DataFrame, threshold=0.9) -> pd.DataFrame:
        """
        Zwraca pary cech o |korelacji| >= threshold, posortowane malejąco.
        Kandydaci do usunięcia przed treningiem ML (redundantna informacja,
        ryzyko multikolinearności w modelach liniowych).
        """
        pairs = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if pd.notna(val) and abs(val) >= threshold:
                    pairs.append(
                        {"feature_1": cols[i], "feature_2": cols[j], "correlation": val}
                    )
 
        result = pd.DataFrame(pairs)
        if result.empty:
            return result
        return (
            result.reindex(result["correlation"].abs().sort_values(ascending=False).index)
            .reset_index(drop=True)
        )

    def find_constant_features(self, df: pd.DataFrame, feature_cols=None, tol=1e-12) -> list:
        """
        Zwraca listę cech o odchyleniu standardowym <= tol (praktycznie
        stałych w badanym oknie czasowym). Takie cechy nie niosą informacji
        rozróżniającej stany i powinny zostać usunięte przed etapem ML
        (modele liniowe mogą się na nich wywrócić przy dzieleniu przez
        wariancję; modele drzewiaste po prostu je zignorują, marnując
        pamięć i czas treningu).
        """
        feature_cols = feature_cols or self._default_feature_cols(df)
        stds = df[feature_cols].std()
        return stds[stds <= tol].index.tolist()
 
    # ------------------------------------------------------------------
    # 7. Redukcja redundancji: zachowaj jedną cechę z każdego "klastra"
    #    silnie skorelowanych cech
    # ------------------------------------------------------------------
    def select_uncorrelated_features(self, corr_matrix: pd.DataFrame, threshold=0.9) -> dict:
        """
        Zachłanna redukcja redundancji: dla każdej pary cech o |korelacji|
        >= threshold (posortowanych malejąco wg siły korelacji) usuwa drugą
        z nich, chyba że już wcześniej usunięta. Efekt: z każdej "grupy"
        silnie skorelowanych cech (np. X_mean_5s / X_ewm_5s / X_lag_1)
        zostaje jeden reprezentant.
 
        Zwraca {"keep": [...], "drop": [...]}. Kolejność kolumn w
        corr_matrix decyduje, która cecha z pary jest "pierwsza" (zostaje) -
        warto więc uporządkować feature_cols tak, by preferowane cechy
        (np. surowe, nieprzesunięte) były na początku.
        """
        pairs = self.top_correlated_pairs(corr_matrix, threshold=threshold)
        to_drop = set()
 
        for _, row in pairs.iterrows():
            f1, f2 = row["feature_1"], row["feature_2"]
            if f1 in to_drop or f2 in to_drop:
                continue
            to_drop.add(f2)
 
        keep = [c for c in corr_matrix.columns if c not in to_drop]
        return {"keep": keep, "drop": sorted(to_drop)}