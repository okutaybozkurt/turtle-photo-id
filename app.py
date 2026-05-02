"""
app.py
======
Deniz Kaplumbağası Photo-ID Sistemi — Streamlit Arayüzü.
Tamamen Birey Tanıma (Photo-ID) Odaklıdır.
"""

from __future__ import annotations

import os
import pandas as pd
import streamlit as st
from PIL import Image

from src.preprocessing import preprocess_for_model
from src.matcher import match_photo
from src.db import get_connection


# ── Sayfa Yapılandırması ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Turtle Photo-ID",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Minimal CSS (Tema Uyumlu, Kaplumbağa Konseptli) ───────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Metric kutularını karta çevir */
  [data-testid="stMetric"] {
    border: 1px solid rgba(15, 118, 110, 0.2);
    border-radius: 12px;
    padding: 16px 24px;
    background-color: rgba(15, 118, 110, 0.03);
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
  }
  
  [data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 600;
  }

  /* Sidebar stil */
  [data-testid="stSidebar"] {
    background-image: linear-gradient(180deg, rgba(15, 118, 110, 0.02) 0%, rgba(15, 118, 110, 0.08) 100%);
    border-right: 1px solid rgba(15, 118, 110, 0.15);
  }

  /* Dosya yükleyici alanı */
  [data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(15, 118, 110, 0.4);
    border-radius: 12px;
    padding: 16px;
    background-color: rgba(15, 118, 110, 0.02);
    transition: background-color 0.2s;
  }
  [data-testid="stFileUploader"]:hover {
    background-color: rgba(15, 118, 110, 0.05);
  }

  /* Bölüm başlıkları */
  .section-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #0f766e;
    margin-bottom: 16px;
    margin-top: 32px;
    border-bottom: 1px solid rgba(15, 118, 110, 0.1);
    padding-bottom: 6px;
  }

  /* Eşleşme satırı */
  .match-row {
    border: 1px solid rgba(15, 118, 110, 0.15);
    border-left: 4px solid #0f766e;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    background-color: rgba(15, 118, 110, 0.02);
  }
  .match-code {
    font-size: 1.2rem;
    font-weight: 600;
  }

  /* Bilgi kutuları */
  .note-box {
    background: rgba(15, 118, 110, 0.05);
    border-left: 3px solid #0f766e;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.9rem;
    margin-top: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ── AI Modelleri ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="AI Modelleri Belleğe Yükleniyor...")
def load_cached_embedding_model():
    from src.model import get_embedding_model
    return get_embedding_model()


# ── Veritabanı Yardımcıları ───────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_db_stats():
    """Veritabanından güncel özet bilgileri çeker."""
    conn = get_connection()
    if not conn:
        return {"turtles": 0, "photos": 0}
    
    cursor = conn.cursor(dictionary=True)
    stats = {}
    try:
        cursor.execute("SELECT COUNT(*) as count FROM turtles")
        stats["turtles"] = cursor.fetchone()["count"]
        
        cursor.execute("SELECT COUNT(*) as count FROM photo_embeddings")
        stats["photos"] = cursor.fetchone()["count"]
    except Exception as e:
        print("DB Stat Error:", e)
    finally:
        cursor.close()
        conn.close()
    return stats

def get_registered_turtles():
    """Tüm bireyleri ve sahip oldukları fotoğraf sayılarını getirir."""
    conn = get_connection()
    if not conn:
        return pd.DataFrame()
    query = """
        SELECT t.internal_code as 'Birey Kodu',
               COUNT(pe.id) as 'Kayitli Fotograf Sayisi',
               t.first_seen_date as 'Gozlem Tarihi',
               t.notes as 'Gozlem Notlari',
               MAX(pe.created_at) as 'Son Guncelleme'
        FROM turtles t
        LEFT JOIN photo_embeddings pe ON t.id = pe.turtle_id
        GROUP BY t.id
        ORDER BY COUNT(pe.id) DESC
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows, columns=['Birey Kodu', 'Kayitli Fotograf Sayisi', 'Gozlem Tarihi', 'Gozlem Notlari', 'Son Guncelleme'])


# ── Sidebar ───────────────────────────────────────────────────────────────

def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div style="padding: 10px 0 24px 0; text-align: center;">
          <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#0f766e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2C8.686 2 6 4.686 6 8v1a6 6 0 0 0 12 0V8c0-3.314-2.686-6-6-6z"/>
            <path d="M4 11h.01M20 11h.01M5.5 15l-2.5 3M18.5 15l2.5 3M8.5 17l-1 4M15.5 17l1 4"/>
            <path d="M9 13v1M15 13v1"/>
          </svg>
          <div style="font-size: 1.3rem; font-weight: 600; margin-top: 12px; color: #0f766e;">
            Turtle Photo-ID
          </div>
          <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 4px;">
            Yapay Zeka Destekli Birey Tanıma
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        page = st.radio(
            "Menü",
            ["Birey Sorgulama", "Yeni Birey Ekle (Torsoo-i)", "Veritabanı (Dashboard)"],
        )

        st.divider()

        # Sistem Durumu
        st.markdown("<div style='font-size:0.75rem;font-weight:600;margin-bottom:12px;color:#0f766e;letter-spacing:1px;'>SİSTEM DURUMU</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;font-size:0.85rem;margin-bottom:12px;">
          <div style="width:8px;height:8px;border-radius:50%;background:#10b981;"></div>
          Feature Extractor Aktif
        </div>
        <div style="display:flex;align-items:center;gap:8px;font-size:0.85rem;margin-bottom:12px;">
          <div style="width:8px;height:8px;border-radius:50%;background:#10b981;"></div>
          MySQL Bağlantısı Stabil
        </div>
        """, unsafe_allow_html=True)
        
        stats = get_db_stats()
        st.markdown(f"""
        <div style="margin-top: 24px; padding: 12px; background: rgba(15,118,110,0.05); border-radius: 8px; font-size: 0.8rem; text-align: center;">
            Sistemde <strong>{stats.get('turtles', 0)}</strong> birey ve<br>
            <strong>{stats.get('photos', 0)}</strong> referans fotoğraf kayıtlı.
        </div>
        """, unsafe_allow_html=True)

    return page


# ── Sayfa: Photo-ID Sorgulama ─────────────────────────────────────────────

def render_photo_id():
    st.markdown("## Birey Sorgulama (Photo-ID)")
    st.markdown(
        "<p style='opacity:0.7;font-size:0.95rem;margin-bottom:24px;'>Yeni çekilen bir deniz kaplumbağası fotoğrafını yükleyin. Sistem, yüz haritasını çıkararak veritabanındaki kayıtlı bireylerle eşleştirme yapar.</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Gözlem Fotoğrafını Yükle", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        return

    col_img, col_space, col_act = st.columns([1.5, 0.5, 2])
    with col_img:
        st.image(uploaded, width='stretch')
    with col_act:
        st.write("") 
        run = st.button("Veritabanında Birey Ara", use_container_width=False, type="primary")

    if not run:
        return

    with st.spinner("Yüz haritası (1280-d embedding) çıkarılıyor ve veritabanı taranıyor..."):
        processed = preprocess_for_model(uploaded)
        cached_model = load_cached_embedding_model()
        results   = match_photo(processed, top_k=3, embedding_model=cached_model)

    if not results:
        st.markdown("""
        <div class="note-box">
          <strong>Sonuç Bulunamadı:</strong> Sistemde kayıtlı hiçbir birey ile eşleşme sağlanamadı.
        </div>
        """, unsafe_allow_html=True)
        return

    best_match = results[0]
    best_sim = best_match["similarity"] * 100

    # Minimum Benzerlik Eşiği (Threshold) Kontrolü: 
    # Örneğin aslan gibi alakasız bir görsel yüklendiğinde benzerlik %40'ın altındaysa engellenir.
    if best_sim < 40:
        st.error("Yüklenen görsel bir deniz kaplumbağasına ait gibi görünmüyor veya veritabanındaki hiçbir kaplumbağa ile anlamlı bir benzerlik kurulamadı.")
        return

    st.markdown("<div class='section-label'>Eşleşme Sonuçları</div>", unsafe_allow_html=True)
    
    if best_sim > 85:
        st.success(f"Yüksek Güvenilirlikli Eşleşme: Bu kaplumbağanın **{best_match['internal_code']}** olma ihtimali çok yüksek.")
    elif best_sim > 65:
        st.info(f"Olası Eşleşme: Bu kaplumbağa **{best_match['internal_code']}** olabilir, gözle teyit önerilir.")
    else:
        st.warning(f"Düşük Benzerlik: Veritabanındaki en yakın birey **{best_match['internal_code']}**, ancak büyük ihtimalle sisteme kayıtlı olmayan yeni bir birey.")

    for i, r in enumerate(results):
        sim_pct = r["similarity"] * 100
        img_col, info_col = st.columns([1, 3])

        with img_col:
            if os.path.exists(r["image_path"]):
                st.image(r["image_path"], width='stretch')

        with info_col:
            rank = ["1.", "2.", "3."][i]
            st.markdown(f"""
            <div class="match-row">
              <div style="font-size:0.75rem;opacity:0.6;margin-bottom:4px;font-weight:600;">{rank} REFERANS BİREY</div>
              <div class="match-code">Birey Kodu: {r['internal_code']}</div>
              <div style="margin-top:16px;">
                <span style="font-size:0.85rem;opacity:0.8;">Cosine Benzerlik Skoru:</span> 
                <span style="font-weight:600;color:#0f766e;font-size:1.1rem;margin-left:4px;">%{sim_pct:.1f}</span>
              </div>
              <div style="background:rgba(15,118,110,0.1);height:6px;border-radius:4px;margin-top:8px;overflow:hidden;">
                <div style="background:#0f766e;height:100%;width:{sim_pct:.1f}%;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ── Sayfa: Dashboard ──────────────────────────────────────────────────────

def render_dashboard():
    st.markdown("## Veritabanı İstatistikleri")
    st.markdown(
        "<p style='opacity:0.7;font-size:0.95rem;margin-bottom:24px;'>Sistemde kayıtlı deniz kaplumbağası bireylerinin genel görünümü.</p>",
        unsafe_allow_html=True,
    )

    stats = get_db_stats()

    st.markdown("<div class='section-label'>Genel Bakış</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    c1.metric("Kayıtlı Birey Sayısı", stats.get('turtles', 0))
    c2.metric("Referans Yüz Haritası (Embedding)", stats.get('photos', 0))

    st.markdown("<div class='section-label'>Kayıtlı Bireyler Kataloğu</div>", unsafe_allow_html=True)
    
    df_turtles = get_registered_turtles()
    if not df_turtles.empty:
        st.dataframe(df_turtles, use_container_width=True, hide_index=True)
    else:
        st.info("Sistemde henüz kayıtlı birey bulunmuyor.")


# ── Sayfa: Yeni Birey Ekle ────────────────────────────────────────────────

def render_add_turtle():
    from src.model import extract_embedding
    from src.matcher import save_embedding
    import os

    st.markdown("## Yeni Birey Ekle (Torsoo-i)")
    st.markdown(
        "<p style='opacity:0.7;font-size:0.95rem;margin-bottom:24px;'>Yeni bir deniz kaplumbağası keşfettiğinizde sisteme kaydedebilirsiniz. Naming/İsimlendirme yeteneği de içerir.</p>",
        unsafe_allow_html=True,
    )

    with st.form("new_turtle_form"):
        col1, col2 = st.columns(2)
        with col1:
            t_code = st.text_input("Birey Kodu veya İsmi (Örn: t401, Caretta-Memo)", max_chars=30)
            obs_name = st.text_input("Gözlemci / Fotoğrafçı Adı", max_chars=100)
            obs_email = st.text_input("Gözlemci E-Posta", max_chars=100)
        with col2:
            import datetime
            obs_date = st.date_input("Gözlem Tarihi", value=datetime.date.today())
            obs_notes = st.text_area("Gözlem Notları / Yorumlar", max_chars=500)

        uploaded_file = st.file_uploader("Kaplumbağa Fotoğrafı Yükle", type=["jpg", "jpeg", "png"])
        
        submitted = st.form_submit_button("Yeni Birey Olarak Kaydet", type="primary")

    if submitted:
        if not t_code.strip():
            st.error("Lütfen kaplumbağa için bir kod veya isim giriniz.")
            return
        if not uploaded_file:
            st.error("Lütfen bir kaplumbağa fotoğrafı yükleyiniz.")
            return

        with st.spinner("Yeni birey kaydediliyor ve embedding çıkarılıyor..."):
            try:
                # Gözlemci notlarını birleştir
                full_notes = f"Gözlemci: {obs_name.strip() if obs_name else 'Bilinmiyor'}\n"
                full_notes += f"E-Posta: {obs_email.strip() if obs_email else 'Bilinmiyor'}\n"
                full_notes += f"Notlar: {obs_notes.strip() if obs_notes else 'Yok'}"

                # Turtles tablosuna kaydet
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("INSERT IGNORE INTO species (code, name_sci, name_tr) VALUES ('unknown', 'Chelonioidea', 'Bilinmeyen Deniz Kaplumbağası')")
                conn.commit()
                
                cursor.execute("SELECT id FROM species WHERE code='unknown'")
                species_id = cursor.fetchone()[0]

                cursor.execute("INSERT INTO turtles (internal_code, species_id, first_seen_date, notes) VALUES (%s, %s, %s, %s)", 
                               (t_code.strip(), species_id, obs_date, full_notes))
                conn.commit()
                t_id = cursor.lastrowid
                cursor.close()
                conn.close()

                # Fotoğrafı diske kaydet
                t_dir = os.path.join("dataset_kaggle", "images", t_code.strip())
                os.makedirs(t_dir, exist_ok=True)
                
                img_path = os.path.join(t_dir, f"{uploaded_file.name}")
                with open(img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Embedding çıkar ve MySQL'e kaydet
                prep_img = preprocess_for_model(uploaded_file)
                model = load_cached_embedding_model()
                emb = extract_embedding(model, prep_img)

                success = save_embedding(t_id, img_path, emb)
                if success:
                    st.success(f"**Tebrikler!** '{t_code}' kodlu yeni birey başarıyla kaydedildi.")
                    st.balloons()
                else:
                    st.error("Embedding veritabanına kaydedilirken bir hata oluştu.")
            except Exception as e:
                st.error(f"Kayıt hatası: {e}")


# ── Router ────────────────────────────────────────────────────────────────

def main():
    page = _render_sidebar()
    if "Sorgulama" in page:  
        render_photo_id()
    elif "Ekle" in page:
        render_add_turtle()
    elif "Veritabanı" in page: 
        render_dashboard()


if __name__ == "__main__":
    main()

