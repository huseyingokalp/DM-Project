"""
Veri Madenciliği ve Uygulamaları
Hüseyin GÖKALP - 2540138002
"""

import subprocess, sys
# mlxtend kurulumu
subprocess.run([sys.executable, "-m", "pip", "install", "mlxtend",
                "--break-system-packages", "-q"], check=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import math, warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris, make_moons, make_blobs, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, BaggingClassifier, IsolationForest)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                             silhouette_score, precision_recall_curve,
                             average_precision_score, accuracy_score)
from sklearn.decomposition import PCA

FIG_DIR = "/figures"
fig_counter = [0]

def savefig(name):
    fig_counter[0] += 1
    path = f"{FIG_DIR}/{fig_counter[0]:02d}_{name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  [Şekil kaydedildi] {path}")

# ─────────────────────────────────────────────────────────────
# 1. KURULUM VE VERİ YÜKLEME
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("BÖLÜM 1 — Kurulum ve Veri Yükleme")
print("=" * 60)

iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris)

print(f'Iris veri kümesi boyutu: {X_iris.shape}')
print(f'Sınıflar: {iris.target_names}')
print(f'Öznitelikler: {iris.feature_names}')
print(f'Eğitim/Test: {len(X_train)}/{len(X_test)}')

# ─────────────────────────────────────────────────────────────
# 2. ID3 KARAR AĞACI — ÖZGÜN UYGULAMA
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 2 — ID3 Karar Ağacı (Özgün Uygulama)")
print("=" * 60)

class myDT:
    def readDatasetFromArray(self):
        dataset = [
            ['OUTLOOK','TEMPERATURE','HUMIDITY','WINDY','PLAY'],
            ['sunny','hot','high','FALSE','no'],
            ['sunny','hot','high','TRUE','no'],
            ['overcast','hot','high','FALSE','yes'],
            ['rainy','mild','high','FALSE','yes'],
            ['rainy','cool','normal','FALSE','yes'],
            ['rainy','cool','normal','TRUE','no'],
            ['overcast','cool','normal','TRUE','yes'],
            ['sunny','mild','high','FALSE','no'],
            ['sunny','cool','normal','FALSE','yes'],
            ['rainy','mild','normal','FALSE','yes'],
            ['sunny','mild','normal','TRUE','yes'],
            ['overcast','mild','high','TRUE','yes'],
            ['overcast','hot','normal','FALSE','yes'],
            ['rainy','mild','high','TRUE','no']
        ]
        self.attributeNames = dataset[0][:-1]
        dataset = dataset[1:]
        self.classes = [row[-1] for row in dataset]
        return [row[:-1] for row in dataset], self.classes, self.attributeNames

    def getEntropy(self, p):
        return -p * math.log2(p) if p != 0 else 0

    def getGain(self, dataset, classes, attribute):
        gain, nDataset = 0, len(dataset)
        values = list(set(row[attribute] for row in dataset))
        for value in values:
            newClasses = [classes[i] for i, row in enumerate(dataset)
                          if row[attribute] == value]
            classValues = list(set(newClasses))
            ent = sum(self.getEntropy(newClasses.count(cv)/len(newClasses))
                      for cv in classValues)
            gain += (len(newClasses)/nDataset) * ent
        return gain

    def create_hierarchy(self, data, classes, attrs, maxlevel=-1, level=0):
        nData, nAttr = len(data), len(attrs)
        newClasses = list(set(classes))
        freq = [classes.count(c) for c in newClasses]
        totalEnt = sum(self.getEntropy(f/nData) for f in freq)
        default = classes[freq.index(max(freq))]
        if nData == 0 or nAttr == 0: return default
        if classes.count(classes[0]) == nData: return classes[0]
        gain = [totalEnt - self.getGain(data, classes, a) for a in range(nAttr)]
        best = gain.index(max(gain))
        tree = {attrs[best]: {}}
        for val in set(row[best] for row in data):
            newData = [row[:best]+row[best+1:] for i,row in enumerate(data)
                       if row[best]==val]
            newCls = [classes[i] for i,row in enumerate(data) if row[best]==val]
            tree[attrs[best]][val] = self.create_hierarchy(
                newData, newCls, attrs[:best]+attrs[best+1:], maxlevel, level+1)
        return tree

    def showHierarchy(self, branch, sep=''):
        if isinstance(branch, dict):
            for key, val in branch.items():
                print(f'{sep}{key}')
                self.showHierarchy(val, sep + '  | ')
        else:
            print(f'{sep} -> ({branch})')

dt = myDT()
data, classes, attrs = dt.readDatasetFromArray()
result = dt.create_hierarchy(data, classes, attrs)
print('=== ID3 Karar Ağacı Hiyerarşisi ===')
dt.showHierarchy(result)

# Bilgi kazancı görselleştirmesi
weather_df = pd.DataFrame({
    'Outlook': ['sunny','sunny','overcast','rainy','rainy','rainy','overcast',
                'sunny','sunny','rainy','sunny','overcast','overcast','rainy'],
    'Temperature': ['hot','hot','hot','mild','cool','cool','cool','mild',
                    'cool','mild','mild','mild','hot','mild'],
    'Humidity': ['high','high','high','high','normal','normal','normal','high',
                 'normal','normal','normal','high','normal','high'],
    'Windy': ['FALSE','TRUE','FALSE','FALSE','FALSE','TRUE','TRUE','FALSE',
              'FALSE','FALSE','TRUE','TRUE','FALSE','TRUE'],
    'Play': ['no','no','yes','yes','yes','no','yes','no',
             'yes','yes','yes','yes','yes','no']
})

def entropy(labels):
    n = len(labels)
    if n == 0: return 0
    counts = pd.Series(labels).value_counts()
    probs = counts / n
    return -sum(p * math.log2(p) for p in probs if p > 0)

total_ent = entropy(weather_df['Play'].tolist())
print(f'\nToplam Entropi H(S) = {total_ent:.4f} bit\n')

info_gains = {}
for attr in ['Outlook', 'Temperature', 'Humidity', 'Windy']:
    weighted_ent = 0
    for val in weather_df[attr].unique():
        subset = weather_df[weather_df[attr] == val]['Play'].tolist()
        weighted_ent += (len(subset)/len(weather_df)) * entropy(subset)
    info_gains[attr] = total_ent - weighted_ent
    print(f'IG({attr}) = {info_gains[attr]:.4f} bit')

best_attr = max(info_gains, key=info_gains.get)
print(f'\nKök düğüm olarak seçilen öznitelik: {best_attr}')

# ID3 Bilgi Kazancı Çubuğu
plt.figure(figsize=(8, 5))
bars = plt.bar(info_gains.keys(), info_gains.values(),
               color=['#e74c3c' if k == best_attr else '#3498db' for k in info_gains],
               edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, info_gains.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.axhline(y=total_ent, color='gray', linestyle='--', alpha=0.7,
            label=f'Toplam Entropi H(S)={total_ent:.4f}')
plt.xlabel('Öznitelik', fontsize=12)
plt.ylabel('Bilgi Kazancı (bit)', fontsize=12)
plt.title('ID3 — Öznitelik Bazlı Bilgi Kazancı\n(Weather Veri Kümesi)',
          fontweight='bold', fontsize=13)
plt.legend()
plt.tight_layout()
savefig("id3_bilgi_kazanci")

# ─────────────────────────────────────────────────────────────
# 3. scikit-learn CART VE AŞIRI UYUM ANALİZİ
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 3 — CART ve Aşırı Uyum Analizi")
print("=" * 60)

dt_sk = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
dt_sk.fit(X_train, y_train)
print(f'Eğitim Doğruluğu: {dt_sk.score(X_train, y_train):.4f}')
print(f'Test Doğruluğu:   {dt_sk.score(X_test, y_test):.4f}')
print('\nAğaç Kuralları:')
print(export_text(dt_sk, feature_names=list(iris.feature_names)))

plt.figure(figsize=(14, 7))
plot_tree(dt_sk, feature_names=list(iris.feature_names),
          class_names=list(iris.target_names),
          filled=True, rounded=True, fontsize=8)
plt.title('Iris — CART Karar Ağacı (Entropi, max_depth=4)', fontweight='bold')
savefig("cart_karar_agaci")

# Derinlik - aşırı uyum
depths = range(1, 21)
train_accs, test_accs = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    train_accs.append(m.score(X_train, y_train))
    test_accs.append(m.score(X_test, y_test))

best_d = list(depths)[np.argmax(test_accs)]
plt.figure(figsize=(10, 5))
plt.plot(list(depths), train_accs, 'bo-', label='Eğitim', linewidth=2)
plt.plot(list(depths), test_accs, 'rs-', label='Test', linewidth=2)
plt.axvline(x=best_d, color='green', linestyle='--', label=f'Optimal d={best_d}')
plt.xlabel('Ağaç Derinliği'); plt.ylabel('Doğruluk')
plt.title('Karar Ağacı: Derinlik vs Aşırı Uyum Analizi', fontweight='bold')
plt.legend(); plt.grid(alpha=0.3)
savefig("derinlik_asiri_uyum")
print(f'Optimal derinlik: {best_d}')

# ─────────────────────────────────────────────────────────────
# 3.5 WISCONSIN GÖĞÜS KANSERİ — AMPİRİK KANIT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 3.5 — Wisconsin Göğüs Kanseri Ampirik Kanıt")
print("=" * 60)

cancer = load_breast_cancer()
print(f'Veri kümesi: {cancer.data.shape[0]} örnek × {cancer.data.shape[1]} öznitelik')
print(f'Sınıflar: {cancer.target_names}')

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=0)
print(f'Eğitim/Test: {len(X_train_c)}/{len(X_test_c)}\n')

print('=== Ampirik Kanıt 1: Sınırsız Derinlik (max_depth=None) ===')
id3_model = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=0)
id3_model.fit(X_train_c, y_train_c)
train_acc = accuracy_score(y_train_c, id3_model.predict(X_train_c))
test_acc  = accuracy_score(y_test_c,  id3_model.predict(X_test_c))
print(f'Eğitim Doğruluğu: {train_acc*100:.2f}%')
print(f'Test Doğruluğu:   {test_acc*100:.2f}%')
print(f'Eğitim-Test açığı (overfit): {(train_acc-test_acc)*100:.2f}%\n')

print('=== Ampirik Kanıt 2: Ön-Budama (max_depth=3) ===')
pruned_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
pruned_model.fit(X_train_c, y_train_c)
pruned_train = accuracy_score(y_train_c, pruned_model.predict(X_train_c))
pruned_test  = accuracy_score(y_test_c,  pruned_model.predict(X_test_c))
print(f'Eğitim Doğruluğu: {pruned_train*100:.2f}%')
print(f'Test Doğruluğu:   {pruned_test*100:.2f}%')
print(f'Sınırsız vs Budanmış Kazancı: +{(pruned_test-test_acc)*100:.2f}%\n')

print('=== SONUÇ ===')
print(f'Sınırsız derinlik:  Eğitim {train_acc*100:.2f}%, Test {test_acc*100:.2f}%')
print(f'Budanmış (d=3):     Eğitim {pruned_train*100:.2f}%, Test {pruned_test*100:.2f}%')
print(f'Ağaç karmaşıklığının kısıtlanması genelleme yeteneğini {(pruned_test-test_acc)*100:.2f}% artırmıştır.')

# Ampirik Kanıt Karşılaştırma Grafiği
categories = ['Eğitim (Sınırsız)', 'Test (Sınırsız)', 'Eğitim (Budanmış)', 'Test (Budanmış)']
values = [train_acc*100, test_acc*100, pruned_train*100, pruned_test*100]
colors = ['#e74c3c', '#e74c3c', '#2ecc71', '#2ecc71']
alphas = [0.9, 0.5, 0.9, 0.5]
plt.figure(figsize=(9, 6))
bars = plt.bar(categories, values, color=colors, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
plt.ylim(88, 102)
plt.ylabel('Doğruluk (%)', fontsize=12)
plt.title('Wisconsin Göğüs Kanseri — Ampirik Aşırı Uyum Kanıtı', fontweight='bold')
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig("ampirik_asiri_uyum_kanit")

# ─────────────────────────────────────────────────────────────
# 4. GAUSSIAN NAIVE BAYES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 4 — Gaussian Naive Bayes")
print("=" * 60)

nb = GaussianNB().fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print(f'Test Doğruluğu: {accuracy_score(y_test, y_pred_nb):.4f}\n')
print(classification_report(y_test, y_pred_nb, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Tahmin'); plt.ylabel('Gerçek')
plt.title('Naive Bayes — Karışıklık Matrisi', fontweight='bold')
plt.tight_layout()
savefig("naive_bayes_karisiklik_matrisi")

# ─────────────────────────────────────────────────────────────
# 5. SVM — KERNEL KARŞILAŞTIRMASI
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 5 — SVM Kernel Karşılaştırması")
print("=" * 60)

X_svm = X_iris[:, 2:4]
X_scaled = StandardScaler().fit_transform(X_svm)
svm_scores = {}
for kernel in ['linear', 'rbf', 'poly']:
    svm = SVC(kernel=kernel, degree=3, random_state=42)
    score = cross_val_score(svm, X_scaled, y_iris, cv=5).mean()
    svm_scores[kernel] = score
    print(f'SVM ({kernel:6s}): CV Doğruluk = {score:.4f}')

plt.figure(figsize=(7, 5))
plt.bar(svm_scores.keys(), svm_scores.values(),
        color=['#3498db', '#e67e22', '#9b59b6'], edgecolor='black', linewidth=0.8)
for i, (k, v) in enumerate(svm_scores.items()):
    plt.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
plt.ylim(0.9, 1.01)
plt.ylabel('CV Doğruluk (5-fold)'); plt.xlabel('Kernel')
plt.title('SVM — Kernel Karşılaştırması (Iris, Petal Features)', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig("svm_kernel_karsilastirma")

# SVM karar sınırı görselleştirmesi
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, kernel in zip(axes, ['linear', 'rbf', 'poly']):
    svm_fit = SVC(kernel=kernel, degree=3, random_state=42, probability=True)
    svm_fit.fit(X_scaled, y_iris)
    h = 0.02
    x_min, x_max = X_scaled[:,0].min()-0.5, X_scaled[:,0].max()+0.5
    y_min, y_max = X_scaled[:,1].min()-0.5, X_scaled[:,1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm_fit.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
    scatter = ax.scatter(X_scaled[:,0], X_scaled[:,1], c=y_iris, cmap='Set1',
                         edgecolors='k', s=30, linewidth=0.5)
    ax.set_title(f'SVM ({kernel})', fontweight='bold')
    ax.set_xlabel('Petal Length (std)'); ax.set_ylabel('Petal Width (std)')
plt.suptitle('SVM Karar Sınırları — Kernel Karşılaştırması', fontweight='bold', fontsize=13)
plt.tight_layout()
savefig("svm_karar_sinirlari")

# ─────────────────────────────────────────────────────────────
# 6. MLP YAPAY SİNİR AĞI
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 6 — MLP Yapay Sinir Ağı")
print("=" * 60)

for arch in [(10,), (50,25), (100,50,25)]:
    mlp = MLPClassifier(hidden_layer_sizes=arch, max_iter=500, random_state=42)
    score = cross_val_score(mlp, X_iris, y_iris, cv=5).mean()
    print(f'MLP {str(arch):15s}: {score:.4f}')

mlp = MLPClassifier(hidden_layer_sizes=(50,25), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
plt.figure(figsize=(8, 4))
plt.plot(mlp.loss_curve_, 'b-', linewidth=2)
plt.xlabel('Epoch'); plt.ylabel('Kayıp')
plt.title(f'MLP (50,25) Eğitim Kayıp Eğrisi — Son Kayıp: {mlp.loss_curve_[-1]:.4f}',
          fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
savefig("mlp_kayip_egrisi")

# ─────────────────────────────────────────────────────────────
# 7. TOPLULUK YÖNTEMLERİ
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 7 — Topluluk Yöntemleri")
print("=" * 60)

models = {
    'Karar Ağacı': DecisionTreeClassifier(random_state=42),
    'Bagging': BaggingClassifier(n_estimators=50, random_state=42),
    'Rastgele Orman': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
}
results = {}
for name, m in models.items():
    scores = cross_val_score(m, X_iris, y_iris, cv=10)
    results[name] = scores
    print(f'{name:18s}: {scores.mean():.4f} ± {scores.std():.4f}')

# Topluluk kutu grafiği
plt.figure(figsize=(10, 6))
data_plot = [v for v in results.values()]
bp = plt.boxplot(data_plot, labels=results.keys(), patch_artist=True,
                 medianprops={'color': 'red', 'linewidth': 2})
colors_bp = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c']
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color); patch.set_alpha(0.7)
plt.ylabel('10-fold CV Doğruluk')
plt.title('Topluluk Yöntemleri — Performans Karşılaştırması', fontweight='bold')
plt.xticks(rotation=15, ha='right'); plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig("topluluk_yontemleri_kutu")

# Ortalama doğruluk çubuğu
means = {k: v.mean() for k, v in results.items()}
plt.figure(figsize=(9, 5))
bars = plt.bar(means.keys(), means.values(),
               color=colors_bp, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, means.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.ylim(0.88, 1.005); plt.ylabel('Ortalama CV Doğruluk')
plt.title('Topluluk Yöntemleri — Ortalama Doğruluk', fontweight='bold')
plt.xticks(rotation=15, ha='right'); plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig("topluluk_yontemleri_ortalama")

# ─────────────────────────────────────────────────────────────
# 8. k-EN YAKIN KOMŞU
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 8 — k-En Yakın Komşu")
print("=" * 60)

print('=== k Değerinin Etkisi ===')
k_vals = [1, 3, 5, 7, 11, 15, 21]
k_scores = []
for k in k_vals:
    knn = KNeighborsClassifier(n_neighbors=k)
    sc = cross_val_score(knn, X_iris, y_iris, cv=10).mean()
    k_scores.append(sc)
    print(f'k={k:2d}: {sc:.4f}')

print('\n=== Uzaklık Metrikleri (k=13) ===')
metric_scores = {}
for metric in ['euclidean', 'manhattan', 'chebyshev']:
    knn = KNeighborsClassifier(n_neighbors=13, metric=metric)
    sc = cross_val_score(knn, X_iris, y_iris, cv=10).mean()
    metric_scores[metric] = sc
    print(f'{metric:12s}: {sc:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(k_vals, k_scores, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('k (Komşu Sayısı)'); axes[0].set_ylabel('10-fold CV Doğruluk')
axes[0].set_title('k-NN: k Değerinin Etkisi', fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].axhline(y=max(k_scores), color='red', linestyle='--', alpha=0.7,
                label=f'Maks: {max(k_scores):.4f}')
axes[0].legend()

axes[1].bar(metric_scores.keys(), metric_scores.values(),
            color=['#3498db', '#e67e22', '#9b59b6'], edgecolor='black', linewidth=0.8)
for i, (k, v) in enumerate(metric_scores.items()):
    axes[1].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
axes[1].set_ylim(0.9, 1.01)
axes[1].set_ylabel('10-fold CV Doğruluk'); axes[1].set_xlabel('Uzaklık Metriği')
axes[1].set_title('k-NN: Uzaklık Metrikleri (k=13)', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
plt.suptitle('k-En Yakın Komşu Analizi', fontweight='bold', fontsize=13)
plt.tight_layout()
savefig("knn_analiz")

# ─────────────────────────────────────────────────────────────
# 9. KÜMELEME (K-Means vs DBSCAN)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 9 — Kümeleme (K-Means vs DBSCAN)")
print("=" * 60)

X_moon, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
X_blob, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=42)

print('=== Elbow Method (Blobs) ===')
inertias, silhouettes = [], []
k_range = range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_blob)
    sil = silhouette_score(X_blob, km.labels_)
    inertias.append(km.inertia_)
    silhouettes.append(sil)
    print(f'k={k}: Inertia={km.inertia_:.1f}, Silhouette={sil:.4f}')

km_m = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_moon)
db_m = DBSCAN(eps=0.2, min_samples=5).fit_predict(X_moon)
print(f'\nAy verisi — K-Means Silhouette: {silhouette_score(X_moon, km_m):.4f}')
print(f'Ay verisi — DBSCAN Silhouette:  {silhouette_score(X_moon, db_m):.4f}')

# Elbow grafiği
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('k (Küme Sayısı)'); axes[0].set_ylabel('Inertia (SSE)')
axes[0].set_title('K-Means Elbow Grafiği', fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].plot(list(k_range), silhouettes, 'rs-', linewidth=2, markersize=8)
axes[1].set_xlabel('k (Küme Sayısı)'); axes[1].set_ylabel('Silhouette Skoru')
axes[1].set_title('Silhouette Skoru vs k', fontweight='bold')
axes[1].grid(alpha=0.3)
plt.suptitle('K-Means Hiperparametre Analizi (Blob Verisi)', fontweight='bold')
plt.tight_layout()
savefig("kmeans_elbow_silhouette")

# K-Means vs DBSCAN ay verisi
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(X_moon[:,0], X_moon[:,1], c=km_m, cmap='Set1', s=20)
ax[0].set_title('K-Means — Ay Şekli', fontweight='bold')
ax[0].set_xlabel('X1'); ax[0].set_ylabel('X2')
ax[1].scatter(X_moon[:,0], X_moon[:,1], c=db_m, cmap='Set1', s=20)
ax[1].set_title('DBSCAN — Ay Şekli', fontweight='bold')
ax[1].set_xlabel('X1'); ax[1].set_ylabel('X2')
plt.suptitle('K-Means vs DBSCAN: Doğrusal Olmayan Şekil Karşılaştırması',
             fontweight='bold', fontsize=13)
plt.tight_layout()
savefig("kmeans_vs_dbscan_ay")

# ─────────────────────────────────────────────────────────────
# 10. REGRESYON
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 10 — Regresyon")
print("=" * 60)

np.random.seed(42)
X_reg = np.sort(5 * np.random.rand(100, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.15, 100)

lr_model = LinearRegression().fit(X_reg, y_reg)
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X_reg)
lr_poly = LinearRegression().fit(X_poly, y_reg)
ridge = Ridge(alpha=1.0).fit(X_poly, y_reg)

print(f'Doğrusal   R²: {lr_model.score(X_reg, y_reg):.4f}')
print(f'Polinom d=5 R²: {lr_poly.score(X_poly, y_reg):.4f}')
print(f'Ridge d=5   R²: {ridge.score(X_poly, y_reg):.4f}')

X_test_reg = np.linspace(0, 5, 200).reshape(-1, 1)
X_test_poly = poly.transform(X_test_reg)

plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, color='gray', s=20, alpha=0.7, label='Veri')
plt.plot(X_test_reg, lr_model.predict(X_test_reg), 'b-', linewidth=2,
         label=f'Doğrusal R²={lr_model.score(X_reg,y_reg):.3f}')
plt.plot(X_test_reg, lr_poly.predict(X_test_poly), 'r-', linewidth=2,
         label=f'Polinom d=5 R²={lr_poly.score(X_poly,y_reg):.3f}')
plt.plot(X_test_reg, ridge.predict(X_test_poly), 'g--', linewidth=2,
         label=f'Ridge d=5 R²={ridge.score(X_poly,y_reg):.3f}')
plt.xlabel('X'); plt.ylabel('y')
plt.title('Regresyon Karşılaştırması: Doğrusal vs Polinom vs Ridge', fontweight='bold')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
savefig("regresyon_karsilastirma")

# ─────────────────────────────────────────────────────────────
# 11. BİRLİKTELİK KURALI MADENCİLİĞİ (APRİORİ)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 11 — Birliktelik Kuralı Madenciliği (Apriori)")
print("=" * 60)

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

transactions = [
    ['ekmek','süt','yumurta'], ['ekmek','süt','peynir','tereyağı'],
    ['süt','peynir','yumurta'], ['ekmek','süt','peynir','yumurta'],
    ['ekmek','tereyağı'], ['süt','yumurta','peynir'],
    ['ekmek','süt','yumurta','peynir'], ['ekmek','süt'],
    ['ekmek','peynir','yumurta','tereyağı'], ['süt','peynir','tereyağı'],
]
te = TransactionEncoder()
df_te = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
freq = apriori(df_te, min_support=0.3, use_colnames=True)
rules = association_rules(freq, metric='confidence', min_threshold=0.5)
print(f'Sık öge kümesi: {len(freq)}, Kural sayısı: {len(rules)}\n')
top5 = rules.nlargest(5, 'lift')[['antecedents','consequents','support','confidence','lift']]
print(top5.to_string())

# Apriori Destek-Güven Dağılımı
plt.figure(figsize=(8, 6))
scatter = plt.scatter(rules['support'], rules['confidence'],
                      c=rules['lift'], cmap='viridis', s=80, alpha=0.8, edgecolors='k', linewidth=0.5)
plt.colorbar(scatter, label='Lift')
plt.xlabel('Destek (Support)', fontsize=12)
plt.ylabel('Güven (Confidence)', fontsize=12)
plt.title('Birliktelik Kuralları — Destek vs Güven (Renk=Lift)', fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
savefig("apriori_destek_guven_dagilimi")

# ─────────────────────────────────────────────────────────────
# 12. AYKIRI DEĞER TESPİTİ (ISOLATION FOREST)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 12 — Aykırı Değer Tespiti (Isolation Forest)")
print("=" * 60)

np.random.seed(42)
X_normal = np.random.randn(200, 2) * 0.8
X_outliers_data = np.random.uniform(low=-4, high=4, size=(15, 2))
X_all = np.vstack([X_normal, X_outliers_data])

iso = IsolationForest(contamination=0.07, random_state=42)
labels = iso.fit_predict(X_all)
scores_if = iso.score_samples(X_all)
print(f'Aykırı değer tespit: {sum(labels==-1)}')
print(f'Normal gözlem:       {sum(labels==1)}')

plt.figure(figsize=(8, 6))
colors_if = ['red' if l == -1 else 'steelblue' for l in labels]
plt.scatter(X_all[:,0], X_all[:,1], c=colors_if, s=40, alpha=0.8,
            edgecolors='k', linewidth=0.4)
from matplotlib.lines import Line2D
legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor='steelblue',
                           markersize=10, label=f'Normal ({sum(labels==1)})'),
                   Line2D([0],[0], marker='o', color='w', markerfacecolor='red',
                           markersize=10, label=f'Aykırı ({sum(labels==-1)})')]
plt.legend(handles=legend_elements, loc='upper right')
plt.xlabel('X1'); plt.ylabel('X2')
plt.title('Isolation Forest — Aykırı Değer Tespiti', fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
savefig("isolation_forest_aykirideğer")

# ─────────────────────────────────────────────────────────────
# 13. MODEL DEĞERLENDİRME (ROC + PR)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 13 — Model Değerlendirme (ROC + PR)")
print("=" * 60)

y_bin = (y_iris == 2).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_bin, test_size=0.3, random_state=42)
classifiers = {
    'Karar Ağacı': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(probability=True, random_state=42),
    'Rastgele Orman': RandomForestClassifier(n_estimators=50, random_state=42),
    'Gradient Boost': GradientBoostingClassifier(n_estimators=50, random_state=42),
}

print('Model             AUC      AP')
print('-' * 40)
roc_data, pr_data = {}, {}
for name, clf in classifiers.items():
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)[:,1]
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_te, y_prob)
    roc_data[name] = (fpr, tpr, roc_auc)
    precision, recall, _ = precision_recall_curve(y_te, y_prob)
    pr_data[name] = (precision, recall, ap)
    print(f'{name:17s} {roc_auc:.4f}   {ap:.4f}')

# ROC Eğrisi
plt.figure(figsize=(8, 7))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')
plt.plot([0,1],[0,1],'k--', linewidth=1, label='Rastgele')
plt.xlabel('Yanlış Pozitif Oranı (FPR)'); plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.title('ROC Eğrisi — 5 Sınıflandırıcı Karşılaştırması', fontweight='bold')
plt.legend(loc='lower right'); plt.grid(alpha=0.3)
plt.tight_layout()
savefig("roc_egrisi")

# PR Eğrisi
plt.figure(figsize=(8, 7))
for name, (precision, recall, ap) in pr_data.items():
    plt.plot(recall, precision, linewidth=2, label=f'{name} (AP={ap:.3f})')
plt.xlabel('Recall (Duyarlılık)'); plt.ylabel('Precision (Kesinlik)')
plt.title('Precision-Recall Eğrisi — 5 Sınıflandırıcı', fontweight='bold')
plt.legend(loc='upper right'); plt.grid(alpha=0.3)
plt.tight_layout()
savefig("precision_recall_egrisi")

# ─────────────────────────────────────────────────────────────
# 14. ÖZNİTELİK ÖNEMİ VE PCA
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 14 — Öznitelik Önemi ve PCA")
print("=" * 60)

rf_full = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_iris, y_iris)
print('=== Öznitelik Önemi (Gini Importance) ===')
feat_imp = sorted(zip(iris.feature_names, rf_full.feature_importances_),
                  key=lambda x: -x[1])
for fname, imp in feat_imp:
    print(f'  {fname:25s}: {imp:.4f}')

pca = PCA(n_components=2).fit(X_iris)
print(f'\n=== PCA Analizi ===')
print(f'PC1 açıklanan varyans: {pca.explained_variance_ratio_[0]*100:.1f}%')
print(f'PC2 açıklanan varyans: {pca.explained_variance_ratio_[1]*100:.1f}%')
print(f'Toplam (2 bileşen):    {sum(pca.explained_variance_ratio_)*100:.1f}%')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# Öznitelik önemi
names_fi = [f[0] for f in feat_imp]
imps_fi  = [f[1] for f in feat_imp]
axes[0].barh(names_fi, imps_fi, color=['#e74c3c','#e67e22','#3498db','#2ecc71'],
             edgecolor='black', linewidth=0.8)
for i, v in enumerate(imps_fi):
    axes[0].text(v + 0.003, i, f'{v:.4f}', va='center', fontweight='bold')
axes[0].set_xlabel('Gini Önemi')
axes[0].set_title('Random Forest — Öznitelik Önemi', fontweight='bold')
axes[0].invert_yaxis(); axes[0].grid(axis='x', alpha=0.3)

# PCA scatter
X_pca = pca.transform(X_iris)
colors_pca = ['#e74c3c','#3498db','#2ecc71']
for i, cls_name in enumerate(iris.target_names):
    mask = y_iris == i
    axes[1].scatter(X_pca[mask,0], X_pca[mask,1], c=colors_pca[i],
                    label=cls_name, s=40, edgecolors='k', linewidth=0.4)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title('PCA — 2D Projeksiyon (Iris)', fontweight='bold')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle('Öznitelik Önemi ve PCA Analizi', fontweight='bold', fontsize=13)
plt.tight_layout()
savefig("oznitelik_onemi_pca")

# ─────────────────────────────────────────────────────────────
# 15. JDM — JOINT DISTANCE MEASURE (Awotunde, 2025)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÖLÜM 15 — Joint Distance Measure (JDM)")
print("=" * 60)

def cosine_similarity(u, v):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def minkowski_distance(u, v, p=2):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    return np.sum(np.abs(u - v) ** p) ** (1/p)

def jdm(u, v, p=2):
    d_md = minkowski_distance(u, v, p)
    cos_theta = cosine_similarity(u, v)
    return d_md * abs(cos_theta - 2)

def joint_similarity(u, v, p=2):
    return 1.0 / (1.0 + jdm(u, v, p))

pairs = [
    ('r,s', [-1, 0], [1, 0]),
    ('s,t', [1, 0],  [4, 0]),
    ('u,v', [8, 0],  [10, 0]),
    ('s,w', [1, 0],  [0, 2]),
    ('t,w', [4, 0],  [0, 2]),
    ('s,z', [1, 0],  [1, 1]),
    ('x,y', [-1, 5], [2, -4]),
]

print(f'{"Vektör":8s} {"Açı°":>7s} {"CSM":>8s} {"EDM":>8s} {"JDM(p=2)":>10s} {"JSM":>8s}')
print('-' * 58)
jdm_rows = []
for name, u, v in pairs:
    cos_t = cosine_similarity(u, v)
    angle = np.rad2deg(np.arccos(np.clip(cos_t, -1, 1)))
    edm = minkowski_distance(u, v, p=2)
    d = jdm(u, v, p=2)
    s = joint_similarity(u, v, p=2)
    jdm_rows.append({'Çift':name,'Açı°':round(angle,2),'CSM':round(cos_t,4),
                     'EDM':round(edm,4),'JDM':round(d,4),'JSM':round(s,4)})
    print(f'{name:8s} {angle:>7.2f} {cos_t:>8.4f} {edm:>8.4f} {d:>10.4f} {s:>8.4f}')

# JDM Tablo Görselleştirmesi
jdm_df = pd.DataFrame(jdm_rows)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x_pos = range(len(jdm_df))
axes[0].bar(x_pos, jdm_df['EDM'], width=0.35, align='center',
            label='EDM', color='#3498db', alpha=0.8, edgecolor='k', linewidth=0.7)
axes[0].bar([x+0.38 for x in x_pos], jdm_df['JDM'], width=0.35, align='center',
            label='JDM', color='#e74c3c', alpha=0.8, edgecolor='k', linewidth=0.7)
axes[0].set_xticks([x+0.19 for x in x_pos])
axes[0].set_xticklabels(jdm_df['Çift'], rotation=20)
axes[0].set_ylabel('Mesafe Değeri'); axes[0].legend()
axes[0].set_title('EDM vs JDM Karşılaştırması', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

axes[1].scatter(jdm_df['Açı°'], jdm_df['JSM'],
                s=100, c='#2ecc71', edgecolors='k', linewidth=0.7, zorder=5)
for _, row in jdm_df.iterrows():
    axes[1].annotate(row['Çift'], (row['Açı°'], row['JSM']),
                     textcoords='offset points', xytext=(5,5), fontsize=9)
axes[1].set_xlabel('Açı (°)'); axes[1].set_ylabel('JSM (Benzerlik Skoru)')
axes[1].set_title('JDM: Açı vs Benzerlik Skoru', fontweight='bold')
axes[1].grid(alpha=0.3)
plt.suptitle('Joint Distance Measure — Awotunde (2025)', fontweight='bold', fontsize=13)
plt.tight_layout()
savefig("jdm_analiz")

# JDM-tabanlı k-NN karşılaştırması
print('\n=== k-NN Karşılaştırması (Iris, k=7, 10-fold CV) ===')
iris_reload = load_iris()
X_std = StandardScaler().fit_transform(iris_reload.data)
y_std = iris_reload.target

def jdm_metric(u, v):
    d_md = np.sum(np.abs(u - v) ** 2) ** 0.5
    norm_u, norm_v = np.linalg.norm(u), np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return d_md * 2
    cos_t = np.dot(u, v) / (norm_u * norm_v)
    return d_md * abs(cos_t - 2)

knn_results = {}
for name, knn in [('Euclidean', KNeighborsClassifier(n_neighbors=7, metric='euclidean')),
                   ('Manhattan', KNeighborsClassifier(n_neighbors=7, metric='manhattan')),
                   ('JDM (Awotunde)', KNeighborsClassifier(n_neighbors=7, metric=jdm_metric))]:
    scores = cross_val_score(knn, X_std, y_std, cv=10)
    knn_results[name] = scores.mean()
    print(f'{name:20s}: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%')

plt.figure(figsize=(7, 5))
bars = plt.bar(knn_results.keys(), [v*100 for v in knn_results.values()],
               color=['#3498db','#e67e22','#9b59b6'], edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, knn_results.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val*100:.2f}%', ha='center', va='bottom', fontweight='bold')
plt.ylim(90, 100)
plt.ylabel('10-fold CV Doğruluk (%)'); plt.xlabel('Mesafe Metriği')
plt.title('k-NN: JDM vs Klasik Metrikler (k=7, Iris)', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig("knn_jdm_metrik_karsilastirma")

# ─────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"TAMAMLANDI — {fig_counter[0]} şekil kaydedildi: {FIG_DIR}")
print("=" * 60)
