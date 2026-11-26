# ZeroWaste SmartRecycle CNN MLOps

Altı sınıflı atık görüntüsü sınıflandırma sistemi. Proje, hazır ImageNet ağırlıklarıyla başlatılan ResNet50 CNN modelini atık veri setine göre fine-tune eder ve MLOps uygulamalarını korur.

## Sınıflar

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Veri seti

Kullanılan veri seti, 2.527 görselden oluşan klasör tabanlı `TrashType_Image_Dataset` veri setidir. Ham veri `data_classification/raw/TrashType_Image_Dataset` altında, eğitim için oluşturulan splitler ise şu yapıda tutulur:

```text
data_classification/split/
├── train/<class_name>/
├── valid/<class_name>/
└── test/<class_name>/
```

Ham veri setini çıkardıktan sonra splitleri oluşturun:

```powershell
poetry run python src/prepare_dataset.py `
  --input data_classification/raw/TrashType_Image_Dataset `
  --output data_classification/split
```

Veri setini doğrulama:

```powershell
poetry run python src/validate_dataset.py
poetry run pytest -q
```

## Kurulum

```powershell
poetry install
```

CUDA kullanılabilir durumdaysa config içindeki `device: cuda` ayarı GPU kullanır; GPU yoksa eğitim otomatik olarak CPU'ya düşer.

## Eğitim

`configs/train.yaml` içindeki ayarlarla ResNet50 transfer learning eğitimi başlatılır:

```powershell
poetry run python src/train.py --config configs/train.yaml
```

En iyi model şu konuma kaydedilir:

```text
models/cnn/best_model.pth
```

Veri setindeki `trash` sınıfı diğer sınıflardan daha az örneğe sahip olduğu için eğitimde weighted sampler ve görüntü artırma kullanılır.

## Değerlendirme

```powershell
poetry run python src/evaluate.py --config configs/train.yaml
```

Değerlendirme sonucu `reports/metrics.json` içine yazılır. Raporda accuracy, macro F1, sınıf bazlı precision/recall/F1 ve confusion matrix bulunur. Nesne tespiti için kullanılan mAP metriği bu CNN sınıflandırma akışında kullanılmaz.

## MLOps

- Veri doğrulama: `src/validate_dataset.py`
- Deterministik veri bölme: `src/prepare_dataset.py`
- Model eğitimi: `src/train.py`
- Model değerlendirme: `src/evaluate.py`
- Otomatik testler: `tests/`
- Veri sürümleme: DVC yapılandırması
- CI/CD: `.github/workflows/`
- Docker doğrulama: `docker-compose.yml`

## Eski YOLOv8 çıktıları

CNN dönüşümü aktif eğitim ve değerlendirme akışından YOLOv8 bağımlılığını kaldırır. Eski YOLO veri seti ve deney çıktıları geri dönüş amacıyla workspace'te tutulabilir; yeni CNN kodu bunlara başvurmaz.

## Takım

- Seher Uğurlu
- Damla Su Özbek

**Ders:** MLOps
