import pandas as pd

RAW_PATH = "data/raw/student-mat.csv"   # istersen student-por.csv yap
OUT_PATH = "data/student.csv"

# UCI dosyaları genelde ; ile ayrılır
df = pd.read_csv(RAW_PATH, sep=";")

# Hedef: final not G3 -> binary target
# Mantık: >= 10 geçti (1), < 10 kaldı (0)
df["target"] = (df["G3"] >= 10).astype(int)

# Hedefe direkt ipucu veren kolonları çıkar (overfitting hilesi olmasın)
# G1 ve G2 ara sınavlar, G3'ü neredeyse direkt taşır -> kaldırıyoruz
drop_cols = ["G3", "G2", "G1"]
df = df.drop(columns=drop_cols)

# Eksik değer var mı? (UCI'da genelde yok)
df = df.dropna()

df.to_csv(OUT_PATH, index=False)
print(f"OK -> {OUT_PATH} | shape={df.shape} | target dist:\n{df['target'].value_counts()}")
