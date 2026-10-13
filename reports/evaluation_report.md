# Model Değerlendirme Raporu

## Mevcut durum

Otomatik değerlendirme kodu hazırdır ve eğitim tamamlandıktan sonra bu dosyayı
gerçek metriklerle günceller:

```powershell
poetry run python src/evaluate.py --model runs/train/waste-yolov8/weights/best.pt
```

Şu an `models/best.pt` dosyası `YOLOv8n-cls` (classification) modelidir.
Projedeki `data/data.yaml` ise `images/labels` yapısında bir nesne tespiti
(detection) veri setidir. Bu nedenle mevcut model ile geçerli başarı metriği
üretilemez; iki model/veri görevi eşleştirilmeden sayı yazılmamıştır.

## Metrikler

| Metrik | Değer |
|---|---:|
| Precision | Üretilmedi |
| Recall | Üretilmedi |
| mAP50 | Üretilmedi |
| mAP50-95 | Üretilmedi |

Detection eğitimi tamamlandığında `evaluate.py` gerçek değerleri ve sınıf bazlı
mAP sonuçlarını otomatik olarak bu rapora ve `reports/metrics.json` dosyasına
yazacaktır.
