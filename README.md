

This notebook integrates multiple functionalities for training, evaluating, and performing interpretability analysis using the Qwen2-VL vision-language model. It provides a unified approach for running multi-task learning experiments, detailed ablation studies, and visualization of model attention through Grad-CAM.

## Contents

- [1. Environment Setup](#1-environment-setup)
- [2. Import Libraries & Define Model/Data Functions](#3-import-libraries--define-modeldata-functions)
- [3. Load Model, Processor & Prompt Pool](#4-load-model-processor--prompt-pool)
- [4. Dataset Loading & Conversion](#5-dataset-loading--conversion)
- [5. Evaluation Callback & Metrics](#6-evaluation-callback--metrics)
- [6. Training & Ablation Experiments](#7-training--ablation-experiments)
- [7. Grad-CAM Visualization](#8-grad-cam-visualization)

## Detailed Explanation

### 1. Environment Setup
Installs all required Python libraries, including:
- `transformers`, `datasets`, `peft`, `evaluate` for modeling and evaluation.
- `qwen-vl-utils` specifically for Qwen models.
- `bitsandbytes` for model quantization.
- `captum`, `pytorch_grad_cam` for visualization and interpretability.
- Metric libraries (`rouge_score`, `bert_score`, `pycocoevalcap`) for evaluation.
- Image processing libraries (`matplotlib`, `pillow`).

### 2. Import Libraries & Define Model/Data Functions
Defines the following key components:
- `Qwen2VLWithClassifier`: a custom model integrating both generation and binary classification tasks.
- Custom collate functions (`collate_fn_train` and `collate_fn_eval`) tailored for handling data input to the Qwen2-VL model efficiently.

### 3. Load Model, Processor & Prompt Pool
- Initializes the Qwen2-VL vision-language model with LoRA adapters and 4-bit quantization (for reduced memory usage).
- Defines a flexible prompt pool to support diverse visual question-answering tasks.

### 4. Dataset Loading & Conversion
- Utilizes streaming data loading from the e-SNLI-VE dataset for efficient memory management.
- Preprocesses raw data into structured Hugging Face datasets suitable for the training pipeline.

### 5. Evaluation Callback & Metrics
- Implements a custom evaluation callback (`LowMemEvalCallback`) that computes and prints detailed performance metrics after each epoch.
- Metrics include:
  - **Text Generation**: ROUGE-L, BLEU, METEOR, CIDEr, BERTScore.
  - **Classification**: Accuracy.

### 6. Training & Ablation Experiments
- Defines comprehensive training arguments tailored for fine-tuning and ablation experiments.
- Employs Hugging Face's `Seq2SeqTrainer` to streamline training, evaluation, and ablation study execution.

### 7. Grad-CAM Visualization
- Provides a Grad-CAM visualization workflow for interpreting model predictions.
- Generates attention heatmaps to highlight model focus areas on input images.

## How to Run
Execute the notebook sequentially in a GPU-enabled environment (Google Colab recommended):

- **Environment Setup**: Install dependencies.
- **Optional Drive Setup**: Configure if using Colab with limited storage.
- **Model/Data Preparation**: Run cells from sections 3–5.
- **Training**: Execute the training cell in section 7.
- **Evaluation**: Automatically conducted via the evaluation callback.
- **Visualization**: Execute Grad-CAM visualization cell post-training.

## Requirements
- GPU-enabled runtime (recommended GPU: NVIDIA T4 or higher).
- Adequate disk space or Google Drive integration for caching model data.



- **README.md** (this detailed description)
- **requirements.txt**: listing all Python dependencies
- **.gitignore**: excluding sensitive or large files
- **License**: specify repository license (e.g., MIT, Apache 2.0)
- **Data and Model Information**: Clearly document dataset sources, model checkpoints, and any additional files used.
- **Example Outputs**: Include screenshots or visualizations to demonstrate expected results clearly.
- **Code Comments**: Ensure comprehensive inline documentation within the notebook for better readability.
- **Usage Instructions**: Explicitly provide step-by-step guidance for setting up and running the notebook.

## Results
- Metrics are clearly displayed at each epoch.
- Grad-CAM visualizations provide intuitive insights into model predictions, enhancing interpretability.
