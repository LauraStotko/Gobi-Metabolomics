import pandas as pd
import scipy.stats as stats
import numpy as np
import os

# Dateipfad festlegen
file_path = "/Users/laura.stotko/Documents/Gobi-Metabolomics/data/raw/postprandial_imputed.csv"
results_dir = "/Users/laura.stotko/Documents/Gobi-Metabolomics/data/results/"

# Sicherstellen, dass der Zielordner existiert
os.makedirs(results_dir, exist_ok=True)

# Datei laden und erste Zeile als Header verwenden
try:
    df = pd.read_csv(file_path, sep=None, engine="python", header=None)
except Exception as e:
    print(f"Fehler beim Laden der Datei: {e}")
    exit()

# Erste Zeile als Spaltennamen setzen und entfernen
df.columns = df.iloc[0]  # Setzt die erste Zeile als Spaltennamen
df = df[1:].reset_index(drop=True)  # Entfernt die erste Zeile aus den Daten

# Sicherstellen, dass numerische Werte korrekt umgewandelt werden (Komma zu Punkt)
df.iloc[:, 4:] = df.iloc[:, 4:].replace({',': '.'}, regex=True)
df.iloc[:, 4:] = df.iloc[:, 4:].apply(pd.to_numeric, errors='coerce')

# Sicherstellen, dass alle Spalten tatsächlich numerisch sind
non_numeric_cols = df.iloc[:, 4:].select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print("Warnung: Nicht-numerische Spalten erkannt und übersprungen:", non_numeric_cols)

# Auswahl der Baseline-Daten für OGTT, SLD und OLTT
df_baseline_ogtt = df[(df['challenge'].astype(str) == 'ogtt') & (df['challenge_time'].astype(str) == '0')]
df_baseline_sld = df[(df['challenge'].astype(str) == 'sld') & (df['challenge_time'].astype(str) == '240')]
df_baseline_oltt = df[(df['challenge'].astype(str) == 'oltt') & (df['challenge_time'].astype(str) == '240')]


# Sicherstellen, dass nur gemeinsame Probanden verwendet werden
def filter_common_subjects(df1, df2):
    common_subjects = set(df1['subject']).intersection(set(df2['subject']))
    df1 = df1[df1['subject'].isin(common_subjects)].sort_values('subject')
    df2 = df2[df2['subject'].isin(common_subjects)].sort_values('subject')
    return df1, df2


df_baseline_sld, df_baseline_ogtt = filter_common_subjects(df_baseline_sld, df_baseline_ogtt)
df_baseline_oltt, df_baseline_ogtt = filter_common_subjects(df_baseline_oltt, df_baseline_ogtt)

# Metaboliten auswählen (alle Spalten außer den ersten vier)
metabolites = df.columns[4:]

# Schwellenwert für Signifikanz nach Multiple Testing Korrektur
p_threshold = 0.000079

# Ergebnisse speichern
results = []

for met in metabolites:
    if met in df_baseline_sld.columns and met in df_baseline_ogtt.columns and met in df_baseline_oltt.columns:
        valid_values_ogtt = pd.to_numeric(df_baseline_ogtt[met], errors='coerce').dropna()
        valid_values_sld = pd.to_numeric(df_baseline_sld[met], errors='coerce').dropna()
        valid_values_oltt = pd.to_numeric(df_baseline_oltt[met], errors='coerce').dropna()

        mean_diff_sld, p_val_sld, mean_diff_oltt, p_val_oltt = np.nan, np.nan, np.nan, np.nan

        if len(valid_values_ogtt) > 1 and len(valid_values_sld) > 1:
            mean_diff_sld = valid_values_sld.mean() - valid_values_ogtt.mean()
            t_stat, p_val_sld = stats.ttest_rel(valid_values_sld, valid_values_ogtt, nan_policy='omit')

        if len(valid_values_ogtt) > 1 and len(valid_values_oltt) > 1:
            mean_diff_oltt = valid_values_oltt.mean() - valid_values_ogtt.mean()
            t_stat, p_val_oltt = stats.ttest_rel(valid_values_oltt, valid_values_ogtt, nan_policy='omit')

        significant = (not np.isnan(p_val_sld) and p_val_sld < p_threshold) or (
                    not np.isnan(p_val_oltt) and p_val_oltt < p_threshold)

        results.append({
            'Metabolite': met,
            'Super_Pathway': df[df['Metabolite'] == met]['super_pathway'].values[
                0] if 'super_pathway' in df.columns else 'N/A',
            'Sub_Pathway': df[df['Metabolite'] == met]['sub_pathway'].values[
                0] if 'sub_pathway' in df.columns else 'N/A',
            'Mean_Diff(SLD_OGTT)': round(mean_diff_sld, 6) if not np.isnan(mean_diff_sld) else 'NaN',
            'pvalue(SLD_OGTT)': round(p_val_sld, 10) if not np.isnan(p_val_sld) else 'NaN',
            'Mean_Diff(OLTT_OGTT)': round(mean_diff_oltt, 6) if not np.isnan(mean_diff_oltt) else 'NaN',
            'pvalue(OLTT_OGTT)': round(p_val_oltt, 10) if not np.isnan(p_val_oltt) else 'NaN',
            'Significant_Response': significant
        })

# Ergebnisse in DataFrame speichern
t_test_results = pd.DataFrame(results)

# Ergebnisse speichern
test_results_path = os.path.join(results_dir, "paired_ttest_results_2.csv")
try:
    t_test_results.to_csv(test_results_path, index=False)
    print(f"T-Test abgeschlossen! Ergebnisse gespeichert unter: {test_results_path}")
except Exception as e:
    print(f"Fehler beim Speichern der Testergebnisse: {e}")
