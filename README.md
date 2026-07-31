This is my folder structure

SIGNPAK-AI/
│
├── README.md
├── requirements.txt
├── requirements-training.txt
├── requirements-others.txt
├── config.py
├── main.py
├── check_gpu.py
├── config.py
├── check_environment.py
│
├── scripts/
|   │
|   |
|   |   
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
│   │
│   ├── Signer_1/
│   ├── Signer_2/
│   ├── Signer_3/
│   ├── Signer_4/
│   ├── Signer_5/
│   ├── raw/
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