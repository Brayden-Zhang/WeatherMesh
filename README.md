# WeatherMesh-2: PyTorch Implementation

> [!NOTE] 
> This repository contains an initial draft implementation of the WeatherMesh-2 model. Design decisions were based on the ([technical blog from WindBorne Systems](https://windbornesystems.com/blog/weathermesh-2-technical-blog)).


## Repository Structure
- `models/`: Model definitions (Encoder, Processor, Decoder, etc.)
  
  TODO: 
- `data/`: Scripts for dataset preparation and preprocessing
- `training/`: Training scripts and configurations
- `utils/`: Utility functions for visualization, evaluation, and debugging



# Notes on the Model

## Model Overview
WeatherMesh-2 (WM-2) is a lightweight and computationally efficient weather prediction model capable of:

- Predicting a wide range of weather variables globally
- Supporting multiple resolutions
- Producing 14+ day forecasts

### Key Features:
- **Input Data**: Utilizes ERA-5 as the primary training dataset, along with additional sources. Inputs include global weather variables on a regular grid.
- **Output**: Forecasts the global weather state, including:
  - Geopotential, temperature, winds, and moisture at 25 vertical levels
  - Surface variables such as 2m temperature, dewpoint, wind speeds, precipitation, solar radiation, and total cloud cover


## Model Architecture
### Encoder-Processor-Decoder
- **Encoder**: Converts high-resolution physical space inputs into a latent space representation using convolutional and transformer layers.
- **Processor**: Performs autoregressive forecasting in latent space using transformer blocks with Neighborhood Attention.
- **Decoder**: Translates latent space back into physical space using upconvolutions and transformers.

### Latent Space
- A (B, D, H, W, C) tensor where:
  - **B**: Batch size
  - **D**: Depth (vertical pressure levels)
  - **H, W**: Spatial dimensions (latitude, longitude)
  - **C**: Latent dimension
- Represents the entire state of the atmosphere globally.

### Neighborhood Attention (NATTEN) 
> *I will try to implement this later*
- Backbone architecture leverages neighborhood attention for efficient locality-based learning.
- Advantages:
  - Improves physics-based learning with localized attention.
  - Lower memory footprint and faster performance compared to SWIN-transformer.


## Training Details
- **Pretraining**: Progressive increase in forecast horizons during training (e.g., from 6 hours to 6 days).
- **Fine-tuning**: Custom encoders for different datasets (e.g., ERA-5, IFS HRES, NOAA GFS).
- **Optimization**:
  - Uses distributed Shampoo optimizer for second-order preconditioning.
  - Cosine annealing schedule for learning rate.



