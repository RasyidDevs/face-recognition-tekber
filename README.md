# Face Recognition Real-Time — InsightFace `buffalo_sc`

Sistem pengenalan wajah berbasis **InsightFace `buffalo_sc`** yang berjalan di **CPU**. Identitas ditentukan menggunakan **Cosine Similarity** antara embedding webcam dan embedding referensi tiap orang.

## Alur sistem

```mermaid
flowchart TD
    A[Ambil foto wajah berwarna] --> B[Dataset per orang]
    B --> C[Deteksi dan alignment wajah — SCRFD]
    C --> D[Ekstraksi fitur — buffalo_sc / ArcFace]
    D --> E[Embedding 512 dimensi per foto]
    E --> F[Mean pooling per orang]
    F --> G[L2 normalization]
    G --> H[Simpan dataset_embed/Nama.npy]

    I[Webcam real-time] --> J[Deteksi dan alignment — SCRFD]
    J --> K[Embedding wajah baru 512 dimensi]
    K --> L[Cosine Similarity dengan semua embedding referensi]
    H --> L
    L --> M{Skor tertinggi >= threshold?}
    M -->|Ya| N[Tampilkan nama dan skor]
    M -->|Tidak| O[Tampilkan Unknown]
```

## Dependencies

The project uses **Python 3.13.9** with the following dependencies:

| Library | Version | Purpose |
|---|---|---|
| Python | 3.13.9 | Programming language |
| OpenCV | 4.13.0.92 | Image processing and webcam capture |
| NumPy | 2.4.2 | Embedding processing and cosine similarity |
| InsightFace | 2.1 | Face detection and feature extraction |
| ONNX Runtime | 1.31.0 | CPU-based model inference |


