import pyaudio
import wave
import os

# Parametreler
FORMAT = pyaudio.paInt16   # Ses formatı (16-bit PCM)
CHANNELS = 1               # Kanal sayısı (mono)
RATE = 44100               # Örnekleme hızı (44.1 kHz)
CHUNK = 1024               # Veri bloğu boyutu
RECORD_SECONDS = 5         # Kayıt süresi (5 saniye)
OUTPUT_FOLDER = "voice_commands"  # Kayıtların kaydedileceği klasör

# Eğer klasör yoksa oluştur
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# PyAudio nesnesi oluştur
p = pyaudio.PyAudio()

# Komutlar
commands = ["volume_up", "volume_down", "mute"]

# Her komut için 20 ses kaydını al
for command in commands:
    print(f"{command} komutları kaydediliyor...")
    
    # 21'den 40'a kadar kaydedilecek
    for i in range(21, 41):
        # Ses kaydını başlat
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print(f"{command} {i}. kayıt yapılıyor...")

        frames = []

        # Kayıt al
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # Kaydı durdur
        stream.stop_stream()
        stream.close()

        # Kayıt dosyasına kaydet
        output_filename = os.path.join(OUTPUT_FOLDER, f"{command}_{i}.wav")
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        print(f"{command} {i}. kayıt kaydedildi: {output_filename}")
    
    print(f"{command} komutları kaydedildi.\n")

# PyAudio nesnesini sonlandır
p.terminate()
print("Tüm kayıtlar tamamlandı.")
