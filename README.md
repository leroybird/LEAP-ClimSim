# LEAP-ClimSim: Physics-Based Climate Simulation using AI

13th place solution (out of 697 teams) for the [LEAP - Atmospheric Physics using AI](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim) competition focused on atmospheric physics prediction.

## Model Architecture

- Transformer-based architecture with vertical attention across 60 atmospheric levels [x-transformers](https://github.com/lucidrains/x-transformers) based implementation.
- ROPE positional embeddings were used in all models.
- Ensemble of 6 models with varying configurations 
- HuberLoss for training stability and performance
- Training done on single/dual NVIDIA 4090 GPUs
- Models trained on complete low-resolution dataset from raw NetCDF files
- A lot of other models were tested, including Unet's and 1D convnext models, but they did not preform as well.

## Ensemble Method

Final predictions generated using LightGBM models trained separately for each target feature, taking predictions from the 6 base models as input.

Note: Earlier versions supported neighbouring cell features but were removed to comply with updated competition rules. The final submission used only direct cell features.

## Code

train.py contains the training loop, while train_gb.py contains the code for fitting the 'ensemble model'. Meanwhile, arch.py contains the majority of the NNs tested during the competition.
