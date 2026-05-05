# 💻 Geliştirici Ajanı (Developer Agent)

## 📋 Görev Tanımı
Yazılım mimarisini kurgulamak, derin öğrenme modellerini entegre etmek, veritabanı şemalarını oluşturmak ve kullanıcı arayüzünü (UI) kodlamak.

## 🛠️ Teknik Uygulama
1. **Model Mimarisi:**
   - Pre-trained `EfficientNetB0` kullanılarak transfer learning uygulandı.
   - Modelin son katmanı kesilerek `GlobalAveragePooling2D` ile 1280 boyutlu embedding vektörleri elde edildi.
2. **Veritabanı Entegrasyonu:**
   - MySQL üzerinde `turtles` ve `photo_embeddings` tabloları optimize edildi.
   - Vektör benzerliği için matematiksel `Cosine Similarity` algoritması Python tarafında implement edildi.
3. **UI/UX Geliştirme:**
   - Streamlit kullanılarak Birey Sorgulama, Yeni Birey Ekle ve Dashboard sayfaları kodlandı.

## ⚙️ Kod Standartları
- **Modülerlik:** Veritabanı işlemleri (`src/db.py`), model işlemleri (`src/model.py`) ve eşleştirme mantığı (`src/matcher.py`) birbirinden ayrıldı.
- **Performans:** Apple M1/M2 cihazlarda GPU çakışmalarını önlemek için `.predict()` yerine doğrudan tensör çağırma metodu kullanıldı.

## 🔧 Çözülen Teknik Sorunlar
- **Sorun:** Fine-tuning sonrası modelin sadece eğitim setine odaklanması (overfitting).
- **Çözüm:** ImageNet ağırlıklarının "zero-shot" yeteneği kullanılarak daha geniş bir genelleme başarısı sağlandı.
