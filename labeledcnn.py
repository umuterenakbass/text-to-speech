import os
import numpy as np
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping

# Özelliklerin bulunduğu klasör
FEATURES_FOLDER = "features"

# Özellik dosyalarını ve etiketleri yükle
features = []
labels = []

# Klasördeki .npy dosyalarını yükle
files = glob.glob(os.path.join(FEATURES_FOLDER, "*.npy"))

# Verileri sınıflara göre gruplamak
for file in files:
    # MFCC özelliklerini yükle
    feature = np.load(file)
    features.append(feature)

    # Dosya adını al, "_1.wav" gibi kısımlardan ayırarak komut adını al
    filename = os.path.basename(file)
    if "mute" in filename:
        label = 0
    elif "volumeup" in filename:
        label = 1
    elif "volumedown" in filename:
        label = 2
    labels.append(label)

# Özellikleri numpy array formatına dönüştür
features = np.array(features)

# Etiketleri one-hot encoding formatına dönüştür
labels = to_categorical(labels, num_classes=3)  # 3 sınıf olduğu için

# Eğitim ve test setlerine ayır (Stratified sampling kullanarak)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Veriyi şekillendirme (samples, time_steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # (samples, time_steps, features)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))  # (samples, time_steps, features)

# CNN Modeli Kurulumu
model = Sequential()

# 1. Konvolüsyonel katman
model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(2))

# 2. Konvolüsyonel katman
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))

# 3. Konvolüsyonel katman
model.add(Conv1D(256, 3, activation='relu'))
model.add(MaxPooling1D(2))

# Düzleştirme katmanı (flatten)
model.add(Flatten())

# Tam bağlantılı (fully connected) katman
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout oranını %50 yapıyoruz

# Çıkış katmanı (3 sınıf için)
model.add(Dense(3, activation='softmax'))  # 3 sınıf

# Modeli derleme
optimizer = Adam(learning_rate=0.001)  # Öğrenme oranını optimize ediyoruz
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Modelin özetini yazdırma
model.summary()

# Erken durdurma (Early Stopping) callback ekleyerek aşırı öğrenmeyi engelleme
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Modeli kaydetme
model.save('cnn_model.h5')  # Modeli .h5 formatında kaydediyoruz

# Modelin başarısını test et
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Doğruluğu: {test_acc}")

# Modelin tahminlerini al
y_pred = model.predict(X_test)

# Gerçek etiketleri ve tahmin edilen etiketleri geri dönüştür
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Sınıflandırma raporunu yazdır
print("Sınıflandırma Raporu:")
print(classification_report(y_test_labels, y_pred_labels))
