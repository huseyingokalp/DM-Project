"""
=============================================================================
Karar Ağaçları Analizi - Sınıflandırma ve Regresyon
=============================================================================
Ders    : 11117BLG014 - Veri Madenciliği ve Uygulamaları
Dönem   : 2025-2026 Bahar - Vize Ödevi
Öğrenci : Hüseyin GÖKALP (2540138002)
Konu    : 2. Ödev – Karar Ağaçları

Bölümler:
  1. Kütüphane ve Bağımlılıklar
  2. Karar Ağacı Sınıflandırma (Iris Veri Seti)
  3. GridSearchCV ile Hiperparametre Optimizasyonu (Budama)
  4. Karar Ağacı Regresyonu (Sentetik Veri)
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 1 – KÜTÜPHANELERİN YÜKLENMESİ
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree,
)
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 2 – KARAR AĞACI SINIFLANDIRMA (IRIS VERİ SETİ)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("BÖLÜM 2 – Karar Ağacı Sınıflandırma (Iris Veri Seti)")
print("=" * 65)

# Veri setinin yüklenmesi ve ayrılması
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=99
)

# Sınıflandırıcı Modelinin Eğitimi
clf = DecisionTreeClassifier(random_state=1)
clf.fit(X_train, y_train)

y_pred_clf = clf.predict(X_test)
acc_clf = accuracy_score(y_test, y_pred_clf)
print(f"Temel Model Doğruluk Skoru : {acc_clf:.4f}")

# Karar Ağacı Görselleştirmesi (Şekil 3 / Şekil 5 karşılığı)
plt.figure(figsize=(18, 8))
plot_tree(
    clf,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title("Şekil 3 – Eğitilmiş Karar Ağacı (Budanmamış, Iris)", fontsize=13)
plt.tight_layout()
plt.savefig("sekil3_karar_agaci_budanmamis.png", dpi=150)
plt.show()
print("→ Görsel kaydedildi: sekil3_karar_agaci_budanmamis.png\n")

# Karmaşıklık Matrisi (Şekil / Confusion Matrix)
cm = confusion_matrix(y_test, y_pred_clf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Karmaşıklık Matrisi – Temel Model (Iris)", fontsize=12)
plt.tight_layout()
plt.savefig("sekil_karmaSiklik_matrisi.png", dpi=150)
plt.show()
print("→ Görsel kaydedildi: sekil_karmaSiklik_matrisi.png\n")


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 3 – GRIDSEARCHCV İLE HİPERPARAMETRE OPTİMİZASYONU (BUDAMA)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("BÖLÜM 3 – GridSearchCV ile Hiperparametre Optimizasyonu (Budama)")
print("=" * 65)

# Grid Search ile Budama ve Parametre Optimizasyonu
param_grid = {
    "max_depth": range(1, 10, 1),
    "min_samples_leaf": range(1, 20, 2),
    "min_samples_split": range(2, 20, 2),
    "criterion": ["entropy", "gini"],
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=1),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train, y_train)

# En iyi modelin çıktısı (Şekil 4 karşılığı)
print(f"En İyi Doğruluk Skoru : {grid_search.best_score_:.4f}")
print(f"En İyi Parametreler   : {grid_search.best_params_}")

# Optimize edilmiş model ile test skoru
best_clf = grid_search.best_estimator_
y_pred_best = best_clf.predict(X_test)
acc_best = accuracy_score(y_test, y_pred_best)
print(f"Test Doğruluk Skoru   : {acc_best:.4f}\n")

# Optimize Edilmiş Ağacın Görselleştirilmesi (Şekil 5 karşılığı)
plt.figure(figsize=(18, 8))
plot_tree(
    best_clf,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title(
    "Şekil 5 – Optimizasyon Sonrası Nihai Karar Ağacı (Iris)", fontsize=13
)
plt.tight_layout()
plt.savefig("sekil5_karar_agaci_optimize.png", dpi=150)
plt.show()
print("→ Görsel kaydedildi: sekil5_karar_agaci_optimize.png\n")


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 4 – KARAR AĞACI REGRESYONU (SENTETİK VERİ)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("BÖLÜM 4 – Karar Ağacı Regresyonu (Sentetik, Doğrusal Olmayan Veri)")
print("=" * 65)

# Sentetik, doğrusal olmayan ve gürültülü veri üretimi (Şekil 6 karşılığı)
np.random.seed(42)
X_reg = np.sort(5 * np.random.rand(200, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.2, X_reg.shape[0])

# Eğitim / Test ayrımı
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Sentetik veri dağılımı görselleştirmesi
plt.figure(figsize=(8, 4))
plt.scatter(X_train_r, y_train_r, color="steelblue", s=20, label="Eğitim")
plt.scatter(X_test_r, y_test_r, color="coral", s=20, label="Test")
plt.title("Şekil 6 – Regresyon Analizi İçin Sentetik Dağılım", fontsize=12)
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("sekil6_sentetik_dagilim.png", dpi=150)
plt.show()
print("→ Görsel kaydedildi: sekil6_sentetik_dagilim.png\n")

# Regresyon Modelinin Eğitimi
regressor = DecisionTreeRegressor(max_depth=4, random_state=42)
regressor.fit(X_train_r, y_train_r)

y_pred_r = regressor.predict(X_test_r)
mse = mean_squared_error(y_test_r, y_pred_r)
print(f"Ortalama Kare Hatası (MSE) : {mse:.4f}\n")

# Tahmin Eğrisi Görselleştirmesi (Şekil 7 karşılığı)
X_plot = np.linspace(X_reg.min(), X_reg.max(), 500).reshape(-1, 1)
y_plot = regressor.predict(X_plot)

plt.figure(figsize=(9, 5))
plt.scatter(X_train_r, y_train_r, color="steelblue", s=20, alpha=0.6, label="Eğitim Verisi")
plt.scatter(X_test_r, y_test_r, color="coral", s=20, alpha=0.8, label="Test Verisi")
plt.plot(X_plot, y_plot, color="forestgreen", linewidth=2, label="DT Regressor (max_depth=4)")
plt.title("Şekil 7 – Karar Ağacı Regresyon Modeli Tahmin Eğrisi", fontsize=12)
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("sekil7_regresyon_tahmin.png", dpi=150)
plt.show()
print("→ Görsel kaydedildi: sekil7_regresyon_tahmin.png\n")

# Regresyon Ağacı Yapısının Görselleştirmesi (Şekil 8 karşılığı)
plt.figure(figsize=(18, 7))
plot_tree(
    regressor,
    filled=True,
    rounded=True,
    fontsize=8,
    feature_names=["X"],
)
plt.title("Şekil 8 – Regresyon Problemi İçin Karar Ağacının Düğüm Yapısı", fontsize=13)
plt.tight_layout()
plt.savefig("sekil8_regresyon_agac_yapisi.png", dpi=150)
plt.show()
print("→ Görsel kaydedildi: sekil8_regresyon_agac_yapisi.png\n")


# ─────────────────────────────────────────────────────────────────────────────
# ÖZET RAPOR
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("ÖZET SONUÇLAR")
print("=" * 65)
print(f"  [Sınıflandırma] Temel Model Test Doğruluğu  : %{acc_clf * 100:.2f}")
print(f"  [Sınıflandırma] GridSearch En İyi CV Skoru  : %{grid_search.best_score_ * 100:.2f}")
print(f"  [Sınıflandırma] Optimize Model Test Doğrul. : %{acc_best * 100:.2f}")
print(f"  [Regresyon]     Ortalama Kare Hatası (MSE)  : {mse:.4f}")
print("=" * 65)
print("Tüm görseller çalışma dizinine kaydedildi.")
