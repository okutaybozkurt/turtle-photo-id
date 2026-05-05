# 🛡️ Güvenlik Uzmanı Ajanı (Security Specialist Agent)

## 📋 Görev Tanımı
Sistemin veri güvenliğini sağlamak, hassas bilgilerin sızmasını önlemek ve kodun güvenli standartlarda yazılmasını denetlemek.

## 🔐 Uygulanan Güvenlik Önlemleri
1. **Secrets Yönetimi:** Veritabanı şifreleri, Kaggle kullanıcı adı ve API anahtarı hiçbir şekilde kaynak koda yazılmamıştır. Tüm bu bilgiler `.env` dosyasında izole edilmiştir.
2. **Git Koruması:** `.gitignore` dosyası yapılandırılarak; `.env`, `venv/`, `dataset_kaggle/` ve büyük model dosyalarının GitHub gibi halka açık platformlara sızması engellenmiştir.
3. **Veritabanı Güvenliği:** MySQL bağlantılarında SQL Injection riskine karşı parametreli sorgular (prepared statements) kullanılması sağlanmıştır.

## 📁 Dosya ve Veri Denetimi
- **Denetim:** `dataset_kaggle/` klasörünün boyutu çok büyük olduğu için bulut depolama yerine lokalde tutulması kararlaştırıldı.
- **Onay:** Sistemin "Public" olarak paylaşılmasında hiçbir güvenlik açığı bulunmadığına dair onay verilmiştir.

## 🚨 Risk Yönetimi
- **Risk:** Kullanıcının alakasız/zararlı dosya yüklemesi.
- **Önlem:** Streamlit dosya yükleyicide dosya tipi (PNG, JPG, JPEG) kısıtlaması getirildi.
