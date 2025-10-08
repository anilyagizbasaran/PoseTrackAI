# OpenCV YOLO Pose Detection - Dokümantasyon

Kalıcı kişi tanıma özellikli gelişmiş pose tespiti ve takip sistemi.

---

## Proje Yapısı

```
opencv_yolo/
├── pose_ultralytics.py      # Ana webcam tespit scripti
├── pose_rtsp.py             # RTSP kamera tespit scripti
├── tracking.py              # Norfair + ReID ile gelişmiş takip
├── person_database.py       # Kalıcı kişi veritabanı (JSON/SQLite)
├── pose_utils.py            # Pose hesaplama ve çizim araçları
├── ui.py                    # UI overlay ve görselleştirme
├── log.py                   # Loglama sistemi
├── camera_rtsp.py           # RTSP kamera yönetimi
├── config_manager.py        # Konfigürasyon dosya yöneticisi
├── config_webcam.yaml       # Webcam konfigürasyon dosyası
├── config_rtsp.yaml         # RTSP kamera konfigürasyonu
└── person_database.json     # Kişi veritabanı (otomatik oluşur)
```

---

## Her Dosyanın Görevi

### Ana Scriptler
- **pose_ultralytics.py** - Webcam pose tespiti ve takip
- **pose_rtsp.py** - RTSP kamera pose tespiti ve takip

### Temel Modüller
- **tracking.py** - Norfair, ReID ve kalıcı veritabanı ile takip yönetimi
- **person_database.py** - Kişi embedding'lerini kalıcı olarak saklar
- **pose_utils.py** - Pose keypoint hesaplamaları ve iskelet çizimi
- **ui.py** - Ekran üstü bilgi gösterimi

### Destek Modülleri
- **camera_rtsp.py** - RTSP stream bağlantı ve yönetimi
- **config_manager.py** - YAML konfigürasyon yükleme ve yönetimi
- **log.py** - Zaman damgalı loglama ve istatistikler

### Konfigürasyon
- **config_webcam.yaml** - Webcam konfigürasyonu (kamera, takip, ReID, performans)
- **config_rtsp.yaml** - RTSP kamera konfigürasyonu (webcam ile veritabanı paylaşır)

---

## Hızlı Başlangıç

### Kurulum
```bash
# Bağımlılıkları yükle
pip install opencv-python ultralytics torch torchvision norfair

# YOLO modeli (ilk çalıştırmada otomatik indirilir)
```

### Webcam Tespiti Çalıştır
```bash
python pose_ultralytics.py
```

### RTSP Tespiti Çalıştır
```bash
# config_rtsp.yaml dosyasına RTSP URL'ini gir
python pose_rtsp.py
```

### Klavye Kontrolleri
- **Q veya ESC** - Çıkış
- **P** - Duraklat/Devam
- **F** - Tam ekran
- **W** - Normal pencere

---

## Konfigürasyon

Tüm ayarlar **config_webcam.yaml** dosyasında - davranışı değiştirmek için bu dosyayı düzenle.

### Önemli Ayarlar

**Kamera**
```yaml
camera:
  source: 0                # 0=Webcam, "rtsp://..."=RTSP, "video.mp4"=Dosya
  resolution: [640, 480]   # Düşük=hızlı, Yüksek=kaliteli
```

**Takip**
```yaml
tracking:
  use_norfair: true        # Gelişmiş takip (önerilen)
  use_reid: true           # Kişi yeniden tanıma
  use_persistent_reid: true # Kalıcı veritabanı (kişileri sonsuza kadar tanı)
```

**Performans**
```yaml
performance:
  track_every_n_frames: 1  # Her N frame'i işle (yüksek=hızlı, az doğru)
```

Tüm parametreler için detaylı açıklamalarla **config_webcam.yaml** dosyasına bakın.

---

## Ana Özellikler

### 1. Kalıcı Kişi Tanıma
Kişiler aynı ID'yi alır, şu durumlarda bile:
- Kameradan çıkıp saatler/günler sonra geri gelirse
- Kıyafet değiştirirse
- Görünümü değişirse

Veritabanı: `person_database.json` (otomatik oluşur ve yönetilir)

### 2. Gelişmiş Takip
- **Norfair**: Kalman filter + hareket tahmini
- **ReID**: Görünüm tabanlı yeniden tanıma
- **OKS Distance**: Pose-aware eşleştirme

### 3. RTSP Desteği
- Çoklu kamera
- Kameralar arası paylaşılan kişi veritabanı
- Bağlantı kopmasında otomatik yeniden bağlanma

---

## Çıktılar

### Ekran Gösterimi
- Kişi sayısı
- Aktif takip ID'leri
- Pose kalite yüzdesi
- Baş yönü
- FPS sayacı

### Dosyalar
- **Video**: `yolo11_object_pose_output.avi` veya `rtsp_pose_output.avi`
- **Loglar**: `logs/yolo_detection.log`
- **Veritabanı**: `person_database.json` (kişi embedding'leri)

---

## Yaygın Ayarlamalar

### Performansı İyileştir
```yaml
camera:
  resolution: [320, 240]   # Düşük çözünürlük
performance:
  track_every_n_frames: 3  # Frame atla
tracking:
  use_reid: false          # ReID gereksizse kapat
```

### Daha İyi Kişi Tanıma
```yaml
tracking:
  persistent_similarity_threshold: 0.85  # Daha sıkı eşleştirme (0.65=esnek, 0.85=sıkı)
  reid_weight: 0.7                       # Görünüme daha fazla ağırlık
  keypoint_weight: 0.3                   # Pose'a daha az ağırlık
```

### RTSP Bağlantı
```yaml
camera:
  source: "rtsp://username:password@ip:port/stream"
  buffer_size: 5           # Kararlılık için daha büyük buffer
```

---

## Sistem Gereksinimleri

- **Python**: 3.8+
- **GPU**: Önerilen (CUDA'lı NVIDIA)
- **RAM**: Minimum 4GB, önerilen 8GB
- **Webcam**: 720p veya üzeri önerilen

---

## İpuçları

1. **Basit Başla**: Önce varsayılan config kullan, sonra ayarla
2. **FPS İzle**: Performans için logları kontrol et
3. **Veritabanı Yedekle**: `person_database.json` düzenli yedekle
4. **Config Testleri**: Kullanım durumuna göre farklı parametreler dene
5. **GPU Kullan**: Mümkünse her zaman GPU kullan (`device: "cuda"`)

---

Detaylı parametre açıklamaları için **config_webcam.yaml** dosyasındaki yorumlara bakın