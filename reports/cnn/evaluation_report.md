# CNN Model Değerlendirme Raporu

## Genel Metrikler

- Accuracy: 0.9297
- Macro F1: 0.9096

## Sınıf Bazlı Sonuçlar

| Sınıf | Precision | Recall | F1 | Görüntü |
|---|---:|---:|---:|---:|
| cardboard | 0.9833 | 0.9672 | 0.9752 | 61 |
| glass | 0.9571 | 0.8816 | 0.9178 | 76 |
| metal | 0.7848 | 1.0000 | 0.8794 | 62 |
| paper | 0.9775 | 0.9667 | 0.9721 | 90 |
| plastic | 0.9710 | 0.9178 | 0.9437 | 73 |
| trash | 0.8824 | 0.6818 | 0.7692 | 22 |

## Not

`trash` sınıfı daha az örneğe sahip olduğu için recall değeri diğer sınıflardan düşüktür.
