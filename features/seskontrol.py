import pyaudio
import numpy as np
import librosa
import librosa.display
import wave
import os
from tensorflow.keras.models import load_model
import subprocess
import time

# Modeli yükleyin
model = load_model('cnn_model.h5')

# Ses kaydını alma fonksiyonu
def record_audio(filename, record_seconds=5, rate=44100, chunk=1024):
    p = pyaudio.PyAudio()

    # Ses kaydını başlat
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
    
    print("Kayda başlıyoruz...")

    frames = []
    for i in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Kaydı durdur
    print("Kayıt tamamlandı...")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Ses dosyasını kaydet
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

# MFCC özniteliklerini çıkarma fonksiyonu
def extract_mfcc(filename):
    y, sr = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

# Ses seviyesi ayarlama fonksiyonu
def set_volume(command):
    if command == 'volume_up':
        # AppleScript ile ses seviyesini 7 birim arttırma
        subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 7)"])  
        print("Ses arttırıldı.")
    elif command == 'volume_down':
        # AppleScript ile ses seviyesini 7 birim azaltma
        subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 7)"])  
        print("Ses kısıldı.")
    elif command == 'mute':
        # AppleScript ile sesi tamamen kapatma
        subprocess.run(["osascript", "-e", "set volume output volume 0"])  
        print("Ses kapatıldı.")

# Test etmek için ses kaydını al
filename = "test_recording.wav"

# Sonsuz döngüde ses komutlarını dinle
while True:
    record_audio(filename)

    # MFCC özniteliklerini çıkar
    mfcc_features = extract_mfcc(filename)
    mfcc_features = np.reshape(mfcc_features, (1, -1, 1))  # Modelin beklediği şekle getirme

    # Modeli tahmin yapacak şekilde kullanma
    prediction = model.predict(mfcc_features)

    # Sınıf etiketlerini ve tahmin sonucunu yazdırma
    sınıf_etiketleri = ['mute', 'volume_up', 'volume_down']
    pred_label = np.argmax(prediction, axis=1)

    # Tahmin edilen sınıfı al
    command = sınıf_etiketleri[pred_label[0]]
    print(f"Modelin tahmini: {command}")

    # Ses seviyesini ayarlama
    set_volume(command)

    # Bir sonraki komut için 1 saniye bekle
    time.sleep(1)
