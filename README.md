# Face Verification

This project verifies whether a given image is a duplicate or belongs to the same person, based on precomputed face embeddings stored in the vector database. 
- **Vector Database**: Stored locally at `data/milvus_demo.db` using Milvus Lite  
- **Model**: FaceNet for embedding generation and Euclidean distance to measure similarity (smaller distance → higher similarity)
- **Preprocessing**: MTCNN for detecting and cropping tight face regions

## 🖥️ Requirements

### ✅ Supported Systems

This project has been tested on the following system configuration:

| Operating System | Chip                              | RAM  | Python Version |
| ---------------- | --------------------------------- | ---- | -------------- |
| macOS 15         | Apple M2 (8-Core CPU, 8-Core GPU) | 8 GB | 3.10           |

## ⚙️ Setup Python Environment

You can set up the environment using either **Conda/Miniconda** or **Python venv**. The following guidelines uses **Miniconda**.

1. Install **Miniconda** using [this guidelines](https://www.anaconda.com/docs/getting-started/miniconda/install#basic-install-instructions).

2. Create and activate a new environment:

   ```bash
   conda create -n face_verify_env python=3.10 -y
   conda activate face_verify_env
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 🔐 Setup Credential File

1. Copy the example file `.env.example` as `.env`:

```bash
cp .env.example .env
```

2. [Optional] Open the `.env` file in a text editor and modify the configuration if needed. For default run, leave it as it is.

```env
...
MILVUS_URI=your-milvus-uri
MILVUS_TOKEN=your-milvus-token
MILVUS_COLLECTION=your-collection-name
...
```

## 🚀 How to Run

### Run the app (backend + frontend)
```bash
make dev
```

### Run the database initialisation
```bash
make db
```

### Run the evaluation
```bash
make eval
```

#### Online evaluation metrics
Processing time (implemented in frontend)
#### Offline evaluation metrics
Accuracy, precision, recall, f1-score (implemented in `make eval`)
```bash [executed at 07/09/2025]
Evaluation Results
------------------
TOP-K: 6
Distance Threshold: 1.0
------------------
Accuracy: 0.9990
Precision: 0.7007
Recall: 0.8819
F1-score: 0.7809
```