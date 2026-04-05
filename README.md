# Amazon Deforestation Prediction Pipeline

A machine learning pipeline for predicting spatially explicit deforestation risk in the Brazilian Amazon, using 22 years of distance-based predictor variables derived from satellite imagery. Developed as a capstone project at [Your University].

---

## Overview

This project trains and evaluates two deep learning architectures — a Convolutional Neural Network (CNN) based on ResNet-18 and a CNN with Long Short-Term Memory (CNN-LSTM) — to predict which 3 × 3 km grid cells in a study area will be deforested in the following year. Models are trained on 2022 MapBiomas deforestation labels and evaluated against 2023 ground truth across two 150 × 150 km Areas of Interest (AOI) in Pará state, Brazil.

The pipeline encodes four human activity variables — distance to past deforestation, pasture expansion, mining activity, and roads — as 22-year multiband raster sequences per grid cell, and learns the spatial and temporal proximity patterns that precede forest clearing. Beyond accuracy metrics, the project introduces a dedicated **frontier detection metric** measuring what fraction of cells that were forested in 2022 and deforested by 2023 each model correctly identified — a distinction with direct implications for conservation prioritisation.

---

## Repository Structure

```
├── pipeline/
│   ├── AmazonDeforestationPipeline.py     # Google Earth Engine data processing
|   |-class CNN_LSTM(nn.Module).py         # Building LSTM model
│   ├── DeforestationCNNPipeline.py        # Unified training pipeline (CNN + CNN-LSTM)
│   ├── DeforestationPredictorLSTM.py      # Inference pipeline
│   ├── VariableDiagnosticAnalysis.py      # Predictor variable profiling
│   └── FireDiagnosticAnalysis.py          # Fire scar diagnostic analysis
│
├── models/
│   ├── CNN_LSTM.py                        # CNN-LSTM model class (top-level for torch.load)
│   └── resnet_modified.py                 # Modified ResNet-18 for multiband input
│
├── notebooks/
│   ├── Capstone_image_dataset.ipynb       # Multiband GeoTIFF data processing
│   ├── Capstone_CNN_modeling.ipynb        # CNN training and evaluation
│   ├── Capstone_image_tests.ipynb         # Window tests and experiments
│   └── Capstone_demo_modeling.ipynb       # Initial autoencoder + XGBoost demo
│
├── data/
│   ├── aoi1_labels_2022.shp               # Training labels for AOI1
│   ├── aoi1_labels_2023.shp               # Evaluation labels for AOI1
│   ├── aoi2_labels_2022.shp               # Training labels for AOI2
│   └── aoi2_labels_2023.shp               # Evaluation labels for AOI2
│
└── README.md
```

> **Note:** Multiband GeoTIFF rasters are not included in this repository due to file size. Instructions for generating them via Google Earth Engine are provided below.

---

## Study Areas

Two 150 × 150 km Areas of Interest within the arc of deforestation in Pará state, Brazil.

**AOI1** — Agricultural expansion frontier, ~7 km from nearest indigenous territory. Contiguous pasture-driven deforestation. Two dynamic predictor variables (deforestation, pasture) + static roads.

```python
AOI1_COORDS = [[
    [-52.492676,-3.348922],
    [-52.492676,-2.021065],
    [-51.020508,-2.021065],
    [-51.020508,-3.348922],
    [-52.492676,-3.348922]
]]

```

**AOI2** — Fragmented landscape, ~3 km from nearest indigenous territory, no navigable water access, active mining sites. Three dynamic predictor variables (deforestation, pasture, mining) + static roads.

```python
AOI2_COORDS = [[
  [-51.314203, -6.883175],
  [-51.314203, -5.552995],
  [-49.831263, -5.552995],
  [-49.831263, -6.883175],
  [-51.314203, -6.883175]
]]

```

---

## Data Sources

| Variable | Source | Coverage | Type |
|---|---|---|---|
| Distance to deforestation | MapBiomas Collection 10 | 2000–2021 | Dynamic (yearly) |
| Distance to pasture | MapBiomas Collection 9/10 | 2000–2021 | Dynamic (yearly) |
| Distance to mining | MapBiomas Collection 10.1 | 2000–2021 | Dynamic (yearly, AOI2 only) |
| Distance to roads | GRIP4 (Meijer et al., 2018) | 2011 | Static |
| Deforestation labels | MapBiomas Collection 10 | 2022, 2023 | Binary per cell |
| Fire scars | MapBiomas Fire Collection 4 | 2001–2022 | Diagnostic only |

All datasets are publicly available and accessed via [Google Earth Engine](https://earthengine.google.com/).

---

## Multiband Raster Structure

Each AOI is represented as a single multiband GeoTIFF where each band encodes the Euclidean distance (in pixels at 30m resolution) from every pixel to the nearest occurrence of a human activity class in a given year.

| AOI | Bands | Composition |
|---|---|---|
| AOI1 | 45 | 22 deforestation + 22 pasture + 1 road |
| AOI2 | 67 | 22 deforestation + 22 pasture + 22 mining + 1 road |

Band index formula: `v * bands_per_variable + t` where `v` = variable index, `t` = year index (0 = year 2000).

---

## Model Architectures

### CNN (ResNet-18)
- Modified input layer accepting 45 or 67 bands instead of 3 RGB channels
- Modified output layer: single neuron for binary classification
- Processes all temporal bands as a flat spatial input — no temporal ordering
- Trained from scratch (no pretrained weights)

### CNN-LSTM
- Reorganises flat bands into a sequence of 22 yearly snapshots
- Shared ResNet-18 backbone extracts 512-dim spatial features per year
- 2-layer LSTM (hidden size 128, dropout 0.3) processes features chronologically
- Temporal attention mechanism learns importance weights across 22 years
- Classifier: 128 → 64 → ReLU → Dropout(0.3) → 1

Supports four input modes via `model_type` parameter:
- `"resnet"` — flat multiband CNN
- `"cnn_lstm"` — raw distance sequences
- `"cnn_lstm_delta"` — year-over-year change sequences
- `"cnn_lstm_combined"` — raw distances + deltas concatenated

---

## Training Configuration

| Parameter | Value |
|---|---|
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Epochs | 10 |
| Train/test split | Spatial block split, 20 km blocks, 80/20 |
| Threshold selection | Youden's J statistic from test set ROC curve |

---

## Usage

### 1. Generate Multiband GeoTIFF (Google Earth Engine)

```python
from pipeline.AmazonDeforestationPipeline import AmazonDeforestationPipeline

pipeline = AmazonDeforestationPipeline(
    coords=AOI1_COORDS,
    aoi_name="AOI1"
)
pipeline.build_multiband_image(years=list(range(2000, 2022)))
pipeline.compute_labels(year=2022)
pipeline.compute_labels(year=2023)
```

### 2. Train a Model

```python
from pipeline.DeforestationCNNPipeline import DeforestationCNNPipeline

pipeline = DeforestationCNNPipeline(
    raster_path="aoi1_multiband.tif",
    shapefile_path="aoi1_labels_2022.shp",
    label_col="deforested",
    model_type="cnn_lstm",       # "resnet" | "cnn_lstm" | "cnn_lstm_delta" | "cnn_lstm_combined"
    num_years=22,
    num_variables=2,             # 2 for AOI1, 3 for AOI2
    has_static_band=True,
    epochs=10
)

pipeline.spatial_split()
pipeline.build_model()
pipeline.train(checkpoint_dir="/path/to/checkpoints")
pipeline.evaluate()
```

### 3. Run Forward Prediction

```python
from pipeline.DeforestationPredictorLSTM import DeforestationPredictorLSTM

predictor = DeforestationPredictorLSTM(
    raster_path="aoi1_multiband_2001_2022.tif",   # shifted 1-year window
    shapefile_path="aoi1_labels_2023.shp",
    model_path="trained_model.pth",
    model_type="cnn_lstm",
    threshold=0.986                                # Youden's J threshold from test set
)

predictor.run_inference()
```

### 4. Run Diagnostic Analyses

```python
from pipeline.VariableDiagnosticAnalysis import VariableDiagnosticAnalysis
from pipeline.FireDiagnosticAnalysis import FireDiagnosticAnalysis

# Variable distance and rate-of-change profiles
diag = VariableDiagnosticAnalysis(
    raster_path="aoi1_multiband.tif",
    predictions_path="aoi1_predictions_2023.shp",
    truth_path="aoi1_labels_2023.shp"
)
diag.plot_temporal_profiles()
diag.compute_rate_of_change()

# Fire scar analysis
fire = FireDiagnosticAnalysis(
    predictions_path="aoi1_predictions_2023.shp",
    truth_path="aoi1_labels_2023.shp",
    fire_path="aoi1_fire_scars.shp"
)
fire.plot_fire_by_category()
```

---

## Key Results

| Model | AOI | Overall Accuracy | Newly Deforested Cells Identified |
|---|---|---|---|
| CNN | AOI1 | 0.95 | 8 / 25 (32%) |
| CNN-LSTM | AOI1 | 0.82 | 16 / 25 (64%) |
| CNN | AOI2 | 0.95 | 3 / 23 (13%) |
| CNN-LSTM | AOI2 | 0.96 | 4 / 23 (17%) |

The CNN learned cumulative proximity — cells distant from deforestation in 2000 that converged over 22 years. The CNN-LSTM learned recent acceleration — cells already close and converging faster in recent years. These signals are mutually exclusive in AOI1, producing largely non-overlapping sets of correctly identified frontier cells.

---

## Requirements

```
Python >= 3.8
torch >= 1.12
torchvision
rasterio
geopandas
earthengine-api
scikit-learn
numpy
pandas
matplotlib
tqdm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Authenticate Google Earth Engine:

```bash
earthengine authenticate
```

---

## Limitations

- Road layer is static (GRIP4, 2011) and does not capture unofficial or new road development
- Grid cell resolution of 3 × 3 km limits precision of individual cell predictions
- Study areas are limited to two 150 × 150 km regions in Pará — generalisation to the full Amazon requires further validation
- Fire scar data is diagnostic only; it is not yet incorporated as a predictor variable
- Small number of newly deforested cells per prediction year (25 in AOI1, 23 in AOI2) limits statistical robustness of frontier detection metrics

---

## Future Directions

- Add fire scar distance as a dynamic predictor variable
- Incorporate dynamic road mapping to capture unofficial road expansion
- Implement combined raw + delta input mode (CNN-LSTM combined) and evaluate against current architectures
- Expand to additional AOIs across the Amazon to test generalisability
- Apply Shapley value analysis to quantify the contribution of each predictor variable to individual predictions

---
