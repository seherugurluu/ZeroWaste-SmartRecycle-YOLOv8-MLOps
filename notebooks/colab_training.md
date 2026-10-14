# Google Colab ile GPU Eğitim Rehberi

Bu rehber, veri setini GitHub'a göndermeden Colab GPU üzerinde eğitim yapmak
içindir. `data/` klasörü DVC ile yönetildiği için Colab'a ayrıca yüklenmelidir.

## 1. Veri setini zip yapma (Windows PowerShell)

Proje klasöründe çalıştırın:

```powershell
Compress-Archive -Path data -DestinationPath data.zip -CompressionLevel Fastest
```

Oluşan `data.zip` dosyasını Colab'a yükleyeceğiz. Bu dosyayı GitHub'a
göndermeyin.

## 2. Colab GPU'yu açma

Colab'da `Runtime > Change runtime type > T4 GPU` seçin.

## 3. Projeyi ve bağımlılıkları hazırlama

Colab hücresinde:

```python
!git clone https://github.com/seherugurluu/ZeroWaste-SmartRecycle-YOLOv8-MLOps.git
%cd ZeroWaste-SmartRecycle-YOLOv8-MLOps
!pip install -q ultralytics pyyaml
!nvidia-smi
```

## 4. Veri setini yükleme

```python
from google.colab import files

uploaded = files.upload()  # Bilgisayardaki data.zip dosyasını seçin
```

```python
!unzip -q data.zip -d .
!python src/validate_dataset.py
```

Doğrulama sonucunda 10.464 görsel ve etiket kontrol edilmelidir.

## 5. GPU eğitimini başlatma

İlk deneme için 10 epoch:

```python
!python src/train.py --config configs/train.yaml --device 0 --epochs 10 --batch 8 --imgsz 640
```

Sonuçlar beklendiği gibiyse final eğitim:

```python
!python src/train.py --config configs/train.yaml --device 0 --epochs 50 --batch 8 --imgsz 640
```

Eğitim sonunda en iyi model genellikle şu konumda olur:

```text
runs/train/waste-yolov8/weights/best.pt
```

## 6. Test ve rapor üretme

```python
!python src/evaluate.py \
  --model runs/train/waste-yolov8/weights/best.pt \
  --data data/data.yaml \
  --split test
```

Rapor dosyaları `reports/metrics.json` ve
`reports/evaluation_report.md` olarak oluşur. Modeli ve raporları indirmek
için:

```python
from google.colab import files

files.download("runs/train/waste-yolov8/weights/best.pt")
files.download("reports/metrics.json")
files.download("reports/evaluation_report.md")
```
