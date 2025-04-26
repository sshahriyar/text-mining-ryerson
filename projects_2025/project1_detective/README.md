# **Project Title:** DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning

## Overview
This project implements the methodology from the paper **DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning**.

It delivers a practical Generative AI application for detecting AI-generated text. The solution consists of modular components for data preprocessing, embedding generation, indexing, and inference—integrated with multi-task and multi-level contrastive learning techniques.

The pipeline includes:
- Pretrained transformer-based embedding generation (e.g., `RoBERTa`)
- `faiss`-based vector indexing and similarity search
- $k$-Nearest Neighbor classification
- Support for Training-Free Incremental Adaptation (TFIA)

---
## Folder Structure
```
/ds8008_detective
├── README.md
├── DeTeCtive_implementation.ipynb  # Main entry point for inference
├── data/                           # Datasets and pretrained models
│   ├── OUTFOX.zip
│   └── OUTFOX_best.pth
├── src/
│   ├── dataset.py                  # Dataset preparation and preprocessing
│   ├── text_embedding.py           # Embedding generation
│   ├── index.py                    # FAISS index management
│   ├── infer.py                    # Inference pipeline
│   ├── OUTFOX_utils.py             # Utilities for OUTFOX dataset
│   └── metrics.py                  # Evaluation metric calculations
└── requirements.txt
```

---
## How to Run the Code
1. Clone the repository:
```
git clone https://github.com/tomatoeggriceX/ds8008_detective.git
cd ds8008_detective
```
2. Install dependencies:
The project requires several specific libraries, which are listed in the `requirements.txt` file. Some key dependencies include:
  - Python (>= 3.8)
  - PyTorch (>= 2.1.0)
  - Transformers (>= 4.0.0)
  - faiss-cpu (for CPU-based indexing in FAISS)
  - lightning (Fabric, for distributed training support)
  - numpy, pandas, scikit-learn, tqdm, matplotlib

  Install with:
  ```
  pip install -r requirements.txt
  ``` 

3. Download datasets
  To download the dataset, you can go to [this link](https://drive.google.com/drive/folders/1FNfSmKFE40FHGBfGjypg_JS2aWO_G6gX) and download `OUTFOX.zip` and save into the `/data` folder
  To download the completed model, you can go to [this link](https://huggingface.co/heyongxin233/DeTeCtive/tree/main) and download `OUTFOX_best.pth`and save into the `/data` folder

3. Run the Notebook

  The main file to run is `DeTeCtive_implementation.ipynb`. 
  You can run the Jupyter Notebook cells to test the code:
  ```
  jupyter notebook DeTeCtive_implementation.ipynb
  ```

  - If running in Colab, update the `path` in the first cell to point to your project files on Google Drive.
  - If running locally, make sure the path variable is empty or correctly points to your working directory.

---
## Outputs
- Classification predictions for each input (human vs. AI-generated)
- Evaluation metrics: `accuracy`, `precision`, `recall`, `f1-score`
- `performance_metrics_subplot.png`: Visualization of performance across different $k$ values

---
## Contact / References
For inquiries or contributions, please contact:
- Derek Liu - derek.liu@toronomu.ca
- Jenny Huang - ziying.huang@torontomu.ca
- Gary Kong - gary.kong@torontomu.ca

Links / references to relevant papers:
- [Original Paper (arXiv)](https://doi.org/10.48550/arXiv.2410.20964)
- [Original GitHub Repository](https://github.com/heyongxin233/DeTeCtive)