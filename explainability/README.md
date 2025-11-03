# Explainability Pipeline

This folder contains the code and resources for running interpretability and explainability analyses on machine learning models, specifically Generalized Additive Models (GAMs) and Multi-Layer Perceptrons (MLPs).

## Folder Structure

```
explainability/
├── interp_pipeline.py       # Main script to run the interpretability pipeline
├── fetch_attributions.py    # Functions to compute attributions for GAM and MLP models
├── ice_analysis.py          # Functions for ICE plots and H-statistic computation
├── visualization.py         # Functions for generating visualizations (e.g., box plots)
├── judge_names.txt          # List of judge names used in the analysis
├── README.md                # Documentation for the explainability folder
```

## Features

1. **Interpretability Pipeline**:

   - Supports both GAM and MLP models.
   - Computes attributions for model predictions.
   - Generates intermediate analysis plots (e.g., ICE plots, H-statistics).

2. **Attribution Analysis**:

   - Uses Captum's `InputXGradient` for MLP models.
   - Custom attribution methods for GAM models.

3. **Visualization**:

   - Generates box plots for attributions categorized by data subsets.
   - Saves plots and results to the specified output directory.

4. **Modular Design**:
   - Code is organized into separate modules for attributions, ICE analysis, and visualization.

## Usage

### 1. Run the Interpretability Pipeline

Use the `interp_pipeline.py` script to run the pipeline.

```bash
python interp_pipeline.py --model <path_to_model> \
                          --model_type <GAM_or_MLP> \
                          --data <path_to_data_pkl> \
                          --output <output_directory> \
                          --logging 1
```

#### Arguments:

- `--model`: Path to the trained model file (e.g., `.pkl` for GAM, `.pt` for MLP).
- `--model_type`: Type of the model (`GAM` or `MLP`).
- `--data`: Path to the input data file (in `.pkl` format).
- `--output`: Path to the directory where results will be saved.
- `--logging`: Set to `1` to enable intermediate analysis plots, or `0` to skip them.

### 2. Example Command

```bash
python interp_pipeline.py --model "../results/full_experiments/main_experiment_results/optimal_model.pt" --model_type MLP --data "data_categorized.pkl" --output "output_explain"
```

### 3. Outputs

- Attributions saved in the input data file under the column `attributions`.
- Plots and categorized attributions saved in the specified output directory.

## Dependencies

- Python 3.8+
- Required Libraries:
  - `torch`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `captum`

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Modules

### 1. `interp_pipeline.py`

The main script to run the interpretability pipeline.

### 2. `fetch_attributions.py`

Contains functions to compute attributions:

- `compute_input_x_gradient_batch`: Computes attributions for MLP models using Captum.
- `gam_interp`: Computes attributions for GAM models.

### 3. `ice_analysis.py`

Contains functions for ICE plots and H-statistic computation:

- `plot_h_statistics`: Generates H-statistics for features.

### 4. `visualization.py`

Contains functions for generating visualizations:

- `box_plot_for_subset`: Creates box plots for attributions categorized by subsets.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. Ensure that your code is well-documented and adheres to the existing coding style.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
