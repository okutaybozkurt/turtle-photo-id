-- ================================================================
-- Deniz Kaplumbağası Photo-ID Projesi — MySQL Kurulum Scripti
-- ================================================================
-- Kullanım: mysql -u root -p < setup_database.sql
-- ================================================================

-- Veritabanı oluştur
CREATE DATABASE IF NOT EXISTS turtle_photoid
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE turtle_photoid;

-- ── Tür kataloğu ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS species (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    code        VARCHAR(50)  NOT NULL UNIQUE,
    name_sci    VARCHAR(100) NOT NULL,
    name_tr     VARCHAR(100),
    description TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── Birey kayıtları ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS turtles (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    internal_code   VARCHAR(50) UNIQUE,
    species_id      INT NOT NULL,
    sex             ENUM('Dişi', 'Erkek', 'Bilinmiyor') DEFAULT 'Bilinmiyor',
    first_seen_date DATE,
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (species_id) REFERENCES species(id)
        ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── Gözlem kayıtları ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS observations (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    turtle_id         INT,
    species_id        INT NOT NULL,
    image_path        VARCHAR(255),
    predicted_species VARCHAR(100),
    confidence_score  FLOAT CHECK (confidence_score BETWEEN 0 AND 1),
    location_name     VARCHAR(150),
    latitude          DECIMAL(9,6),
    longitude         DECIMAL(9,6),
    observed_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    observer_notes    TEXT,
    FOREIGN KEY (turtle_id)  REFERENCES turtles(id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    FOREIGN KEY (species_id) REFERENCES species(id)
        ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── Photo-ID embedding vektörleri ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS photo_embeddings (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    turtle_id      INT NOT NULL,
    observation_id INT,
    image_path     VARCHAR(255) NOT NULL,
    embedding      LONGBLOB NOT NULL,
    model_version  VARCHAR(50) DEFAULT 'mobilenetv2_v1',
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (turtle_id)      REFERENCES turtles(id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (observation_id) REFERENCES observations(id)
        ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── Başlangıç verileri: Tür kataloğu ──────────────────────────────────────
INSERT IGNORE INTO species (code, name_sci, name_tr, description) VALUES
(
    'caretta_caretta',
    'Caretta caretta',
    'Caretta Kaplumbağası',
    'Akdeniz''de en yaygın deniz kaplumbağası türü. '
    'Turuncu-kahverengi rengi ve büyük kafasıyla tanınır.'
),
(
    'chelonia_mydas',
    'Chelonia mydas',
    'Yeşil Deniz Kaplumbağası',
    'Tropikal ve subtropikal sularda yaşar. '
    'İsmini yağ dokusundaki yeşil pigmentten alır.'
),
(
    'eretmochelys_imbricata',
    'Eretmochelys imbricata',
    'Hawksbill Kaplumbağası',
    'İnce, sivri gagasıyla diğer türlerden ayrılır. '
    'Kritik derecede nesli tehlike altındadır.'
);

-- ── Özet ──────────────────────────────────────────────────────────────────
SELECT 'Veritabanı kurulumu tamamlandı!' AS durum;
SELECT COUNT(*) AS tur_sayisi FROM species;
