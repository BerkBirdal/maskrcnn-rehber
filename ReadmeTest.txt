# Mask R-CNN Kurulum Rehberi 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.3.0](https://img.shields.io/badge/tensorflow-2.3.0-orange.svg)](https://github.com/tensorflow/tensorflow/releases/tag/v2.3.0)

RTX ve GTX serisi ekran kartları için kapsamlı Mask R-CNN kurulum rehberi.

## 📋 İçindekiler
- [Mask R-CNN Nedir?](#mask-r-cnn-nedir)
- [Gereksinimler](#gereksinimler)
- [RTX Ekran Kartı için Kurulum](#rtx-ekran-kartı-için-kurulum)
- [GTX Ekran Kartı için Kurulum](#gtx-ekran-kartı-için-kurulum)
- [Ortam Kurulumu](#ortam-kurulumu)
- [COCO API Kurulumu](#coco-api-kurulumu)
- [Model Testi](#model-testi)
- [Yaygın Sorunlar ve Çözümleri](#yaygın-sorunlar-ve-çözümleri)
- [Uygulama Alanları](#uygulama-alanları)

## ℹ️ Mask R-CNN Nedir?

Mask R-CNN, nesne tespiti ve segmentasyon problemleri üzerine geliştirilmiş, ResNet101 ve Özellik Piramit Ağı (FPN) tabanlı bir derin öğrenme modelidir. Bu model:

- Nesne tespiti
- Nesne segmentasyonu
- Görüntü analizi
- Gerçek zamanlı işleme

gibi alanlarda kullanılmaktadır.

## 🔧 Gereksinimler

### Gerekli Yazılımlar
- [Anaconda](https://www.anaconda.com/download/success) - Python ortam yönetimi için
- [Git](https://git-scm.com/downloads) - Kod deposu yönetimi için
- [Python](https://www.python.org/downloads/) - Programlama dili
- [Microsoft Build Tools 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48159) - C++ derleme araçları
- [Rustup](https://www.rust-lang.org/tools/install) - Rust dili araç zinciri
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/?q=build+tools) - C++ yapı araçları

### Gerekli Dosyalar
- [mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases) - Önceden eğitilmiş ağırlıklar

## 💻 RTX Ekran Kartı için Kurulum

### CUDA ve cuDNN Kurulumu
1. [CUDA 11.2](https://developer.nvidia.com/cuda-downloads) kurulumu
2. [cuDNN 8.1.0](https://developer.nvidia.com/rdp/cudnn-archive) kurulumu
3. Aşağıdaki komutu çalıştırın:
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

### Kod Deposu Kurulumu
```bash
git clone https://github.com/leekunhee/Mask_RCNN.git
cd Mask_RCNN
```

### Ortam Kurulumu
```bash
conda create -n MaskR python=3.8
conda init
# Terminali yeniden başlatın
conda activate MaskR
```

### Paket Kurulumu
```bash
pip install tensorflow==2.3.0
pip install protobuf==3.20.3
pip install opencv-python==4.3.0.38
```

### requirements.txt İçeriği
```txt
numpy==1.18.5
scipy==1.4.1
Pillow==10.4.0
keras==2.15.0
cython==0.29.21
matplotlib==3.2.2
scikit-image==0.19.3
h5py==2.10.0
imgaug==0.4.0
IPython==7.34.0
```

## 💻 GTX Ekran Kartı için Kurulum

### CUDA ve cuDNN Kurulumu
1. [CUDA 9.0](https://developer.nvidia.com/cuda-90-download-archive) kurulumu
2. [cuDNN 7.0.5](https://developer.nvidia.com/rdp/cudnn-archive) kurulumu

### Maritime Mask R-CNN Kod Deposu Kurulumu
```bash
git clone https://github.com/Allopart/Maritime_Mask_RCNN.git
cd Maritime_Mask_RCNN
```

### Ortam Kurulumu
```bash
conda create -n MaskR python=3.6
conda init
# Terminali yeniden başlatın
conda activate MaskR
```

### Paket Kurulumu
```bash
pip install tensorflow-gpu==1.5
pip install opencv-python==4.3.0.36
pip install keras==2.1.5
pip install h5py==2.10.0
```

## 📦 COCO API Kurulumu

1. COCO API'yi klonlayın:
```bash
git clone https://github.com/philferriere/cocoapi.git
```

2. Python API'yi yükleyin:
```bash
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

## 🧪 Model Testi

1. Jupyter Notebook'u başlatın:
```bash
jupyter notebook
```

2. `samples` klasörüne gidin
3. `demo.ipynb` dosyasını açın
4. Her hücreyi `Shift + Enter` ile çalıştırın

## ❗ Yaygın Sorunlar ve Çözümleri

1. **CUDA Bulunamadı Hatası**
   - CUDA'nın sistem PATH'inde olduğundan emin olun
   - `nvidia-smi` komutu ile CUDA kurulumunu doğrulayın

2. **Import Error: DLL load failed**
   - Visual C++ Redistributable'ı yeniden yükleyin
   - Python ve CUDA uyumluluğunu kontrol edin

3. **Bellek Hatası**
   - Batch size'ı düşürün
   - Gereksiz uygulamaları kapatın

## 🎯 Uygulama Alanları

Mask R-CNN aşağıdaki alanlarda yaygın olarak kullanılmaktadır:

1. **Otonom Araçlar**
   - Nesne tespiti ve ayrıştırma
   - Yol ve trafik analizi

2. **Medikal Görüntüleme**
   - Organ ve lezyon segmentasyonu
   - Tümör tespiti

3. **Uydu Görüntüleme**
   - Coğrafi nesne analizi
   - Arazi sınıflandırma

4. **Endüstriyel Uygulamalar**
   - Kalite kontrol
   - Üretim hattı izleme

## 🤝 Katkıda Bulunma

1. Bu depoyu forklayın
2. Özellik dalınızı oluşturun (`git checkout -b özellik/YeniÖzellik`)
3. Değişikliklerinizi commitleyin (`git commit -m 'Yeni özellik eklendi'`)
4. Dalınıza push yapın (`git push origin özellik/YeniÖzellik`)
5. Bir Pull Request oluşturun

## 📝 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- Orijinal Mask R-CNN implementasyonu için [Matterport](https://github.com/matterport/Mask_RCNN)'a
- [COCO API](https://github.com/cocodataset/cocoapi) ekibine
- Test ve dokümantasyonda yardımcı olan tüm katkıda bulunanlara

---

❗ **Not**: Bu kurulum rehberi düzenli olarak güncellenmektedir. Herhangi bir sorun yaşarsanız veya güncellemelerle ilgili bilgi almak için issues sekmesini kontrol edebilirsiniz.
