import os
import numpy as np
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from collections import defaultdict

# Özelliklerin bulunduğu klasör
FEATURES_FOLDER = "features"

# Özellik dosyalarını ve etiketleri yükleme
features = []
labels = []

# Klasördeki .npy dosyalarını yükle
files = glob.glob(os.path.join(FEATURES_FOLDER, "*.npy"))

# Verileri sınıflara göre gruplamak
class_groups = defaultdict(list)

for file in files:
    # MFCC özelliklerini yükle
    feature = np.load(file)
    features.append(feature)

    # Dosya adını al, "_1.wav" gibi kısımlardan ayırarak komut adını al
    filename = os.path.basename(file)
    label = filename.split("_")[0]  # "volume_up_1.wav" -> "volume_up"
    
    labels.append(label)
    
    # Sınıfa göre dosyayı gruplama
    class_groups[label].append(feature)

# Özellikleri numpy array formatına dönüştür
features = np.array(features)

# Etiketleri sayısal formata dönüştürmek için LabelEncoder kullan
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Etiketleri one-hot encoding formatına dönüştür
labels = to_categorical(labels, num_classes=3)  # 3 sınıf olduğu varsayılmış

# Eğitim ve test verisi için dizileri başlat
X_train, y_train, X_test, y_test = [], [], [], []

# Sınıf başına 6 örnek test için ayırıp, geri kalanları eğitim için kullan
for label, samples in class_groups.items():
    num_samples = len(samples)
    test_size = 12
    train_size = num_samples - test_size

    # Test seti için 6 örnek
    X_test.extend(samples[:test_size])
    y_test.extend([label] * test_size)

    # Eğitim seti için geri kalan örnekler
    X_train.extend(samples[test_size:])
    y_train.extend([label] * train_size)

# Verileri numpy array formatına dönüştür
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Etiketleri one-hot encoding formatına dönüştür
y_train = to_categorical(label_encoder.transform(y_train), num_classes=3)
y_test = to_categorical(label_encoder.transform(y_test), num_classes=3)

# Veri Şekillendirme: CNN için 3D format (samples, time_steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # (samples, time_steps, 1 feature)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))  # (samples, time_steps, 1 feature)

# CNN Modeli Kurulumu
model = Sequential()

# 1. Convolutional katman
model.add(Conv1D(128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# 2. Convolutional katman
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# 3. Convolutional katman
model.add(Conv1D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Flatten katmanı (2D'den 1D'ye geçiş)
model.add(Flatten())

# Tam bağlantılı (fully connected) katman
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Çıkış katmanı (3 sınıf için)
model.add(Dense(3, activation='softmax'))  # 3 sınıf

# Modeli derleme
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Modeli kaydetme
model.save('cnn_model.h5')

# Modelin başarısını test et
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Doğruluğu: {test_acc}")

# Modelin tahminlerini al
y_pred = model.predict(X_test)

# Gerçek etiketleri ve tahmin edilen etiketleri geri dönüştür
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# F1 skoru, precision, recall, accuracy gibi metrikleri yazdır
print("Sınıflandırma Raporu:")
print(classification_report(y_test_labels, y_pred_labels))

for i, sample in enumerate(X_train[:5]):
    print(f"Eğitim örneği {i}: Etiket -> {y_train[i]}")

