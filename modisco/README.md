# TF-MoDISco Analysis for TargetGAN

This directory contains the workflow for interpreting the TargetGAN model using **DeepLIFT** and **TF-MoDISco**. This analysis helps identifying important sequence motifs that drive the model's predictions.

## 🛠️ Environment Setup

Using DeepLIFT and TF-MoDISco (v0.5.1.1) requires a specific Python environment with TensorFlow 1.x. Please creating a dedicated conda environment using the provided YAML file:

```bash
conda env create -f deeplift_environment.yml
conda activate deeplift
```

**Note:** Do not attempt to run this analysis in the main `TargetGAN` environment, as it relies on incompatible library versions.

## 🚀 Running the Analysis

The complete analysis pipeline is implemented in the Jupyter Notebook:

**[combined_workflow.ipynb](./combined_workflow.ipynb)**

### Steps covered in the notebook:
1.  **Improtace Scoring**: Loads the trained predictor and calculates DeepLIFT contribution scores for promoter sequences.
2.  **Motif Discovery**: Runs TF-MoDISco to cluster high-importance seqlets into motifs.
3.  **Visualization**: Displays the discovered motifs (PWMs) and their statistics.

## 📂 Files

*   `combined_workflow.ipynb`: The main notebook execution script.
*   `deeplift_environment.yml`: Conda environment specification file.
*   `*.meme`: Generated motif files (output) compatible with MEME suite tools.
