This is my folder structure

SIGNPAK-AI/
│
├── README.md
├── requirements.txt
├── requirements-training.txt
├── requirements-others.txt
├── config.py
├── main.py
│
│
├── scripts/
│   ├── 01_download_dataset.py
│   ├── 02_validate_dataset.py
│   ├── 03_preprocess_videos.py
│   ├── 04_extract_landmarks.py
│   ├── 05_augment_dataset.py
│   ├── 06_generate_metadata.py
│   ├── 07_create_csv_splits.py
│   └── 08_train_model.py
|   │
|   ├── check_gpu.py
|   │
|   ├── check_environment.py
│
├── src/
│   ├── downloader.py
│   ├── validator.py
│   ├── preprocess.py
│   ├── mediapipe_utils.py
│   ├── augmentation.py
│   ├── metadata.py
│   ├── csv_utils.py
│   └── utils.py
│
├── data/
│   │
│   ├── raw/
│   │   ├── Greetings/
│   │   │   └── original/
│   │   │       ├── hello.mp4
│   │   │       ├── goodbye.mp4
│   │   │       └── ...
│   │   │
│   │   └── Common_Expressions/
│   │       └── original/
│   │           ├── yes.mp4
│   │           ├── no.mp4
│   │           └── ...
│   │
│   ├── repetitions/
│   ├── cropped/
│   ├── processed/
│   ├── landmarks/
│   ├── augmented/
│   ├── logs/
│   │
│   ├── metadata/
│   │   ├── categories/
│   │   │   ├── greetings.json
│   │   │   └── common_expressions.json
│   │   │
│   │   ├── master/
│   │   │   ├── categories.json
│   │   │   ├── master_dataset.json
│   │   │   ├── master_dataset_clean.json
│   │   │   ├── landmarks_metadata.json
│   │   │   └── augmented_metadata.json
│   │   │
│   │   └── reports/
│   │       ├── validation_report.csv
│   │       └── download_log.csv
│   │
│   └── csv/
│       ├── labels.csv
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
└── models/
    ├── checkpoints/
    └── results/