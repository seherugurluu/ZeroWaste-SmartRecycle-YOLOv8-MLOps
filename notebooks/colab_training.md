# Google Colab ile ResNet50 CNN Eğitimi

Yerel bilgisayarda GPU yoksa CNN eğitimini Google Colab GPU üzerinde çalıştırın.

## 1. Colab GPU'yu açma

Colab'da `Runtime > Change runtime type > T4 GPU` seçin.

## 2. Projeyi ve veri setini yükleme

Projeyi GitHub'dan klonlayın ve çalışma klasörüne geçin:

```python
!git clone https://github.com/seherugurluu/ZeroWaste-SmartRecycle-YOLOv8-MLOps.git
%cd ZeroWaste-SmartRecycle-YOLOv8-MLOps
!pip install -q torch torchvision pillow pyyaml scikit-learn
!nvidia-smi
```

Bilgisayarınızdaki `new_dataset.zip` dosyasını Colab'a yükleyin:

```python
from google.colab import files
uploaded = files.upload()
```

Zip dosyasını ham veri klasörüne çıkarın:

```python
!mkdir -p data_classification/raw
!unzip -q new_dataset.zip -d data_classification/raw
```

## 3. Train/validation/test split oluşturma

```python
!python src/prepare_dataset.py \
  --input data_classification/raw/TrashType_Image_Dataset \
  --output data_classification/split
!python src/validate_dataset.py
```

Beklenen toplamlar yaklaşık olarak şöyledir:

```text
train: 1766
valid: 377
test: 384
```

## 4. CNN eğitimini başlatma

İlk deneme için `configs/train.yaml` içinde `epochs` değerini 3 veya 5 yapabilirsiniz. GPU üzerinde eğitim:

```python
!python src/train.py --config configs/train.yaml
```

Model şu konuma kaydedilir:

```text
models/cnn/best_model.pth
```

## 5. Test ve rapor üretme

```python
!python src/evaluate.py --config configs/train.yaml
```

Raporu görüntülemek için:

```python
import json
with open("reports/metrics.json", encoding="utf-8") as file:
    print(json.dumps(json.load(file), indent=2))
```

Modeli ve raporu bilgisayara indirmek için:

```python
files.download("models/cnn/best_model.pth")
files.download("reports/metrics.json")
```

## Notlar

- Bu rehber ResNet50 CNN içindir; YOLOv8 komutları kullanılmamalıdır.
- `trash` sınıfı az örnekli olduğu için eğitim kodu weighted sampler ve augmentation kullanır.
- İlk eğitimde `epochs: 3` veya `epochs: 5` ile kontrol yapıp, sonuç uygunsa `epochs: 20` ile final eğitime geçin.
