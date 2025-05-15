import librosa
import numpy as np
import os
import glob
import librosa.display
import matplotlib.pyplot as plt

# Kayıt dosyalarının bulunduğu klasör
INPUT_FOLDER = "voice_commands"
OUTPUT_FOLDER = "features"

# Eğer özellikler için klasör yoksa oluştur
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Ses dosyalarını bul
files = glob.glob(os.path.join(INPUT_FOLDER, "*.wav"))

# MFCC'yi çıkartıp kaydetmek için bir fonksiyon
def extract_mfcc(filename):
    # Ses dosyasını yükle
    y, sr = librosa.load(filename, sr=None)  # sr=None, orijinal örnekleme hızını korur

    # MFCC özniteliklerini çıkar
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50)
    
    # Öznitelikleri birleştirerek tek bir vektör haline getir
    mfcc_mean = np.mean(mfcc, axis=1)
    
    return mfcc_mean

# Her ses dosyası için MFCC özniteliklerini çıkar ve kaydet
for file in files:
    filename = os.path.basename(file)
    print(f"{filename} dosyasından MFCC çıkarılıyor...")
    
    # MFCC özniteliklerini çıkar
    mfcc_features = extract_mfcc(file)
    
    # Öznitelikleri dosyaya kaydet (örn. volume_up_1.txt)
    output_filename = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.npy")
    np.save(output_filename, mfcc_features)

    print(f"{filename} için MFCC öznitelikleri kaydedildi: {output_filename}")

print("Tüm MFCC öznitelikleri çıkarıldı ve kaydedildi.")
