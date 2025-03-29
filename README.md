
# GraCo: Towards GRammar-Assisted COunterfactuals

GraCo is a novel Counterfactual Generation (CG) method that integrates feature domain and user information by using Grammatical Evolution to produce plausible and actionable counterfactuals.

## Description

- We introduce a counterfactual generation (CG) method that integrates feature domain and user information to produce credible and actionable counterfactuals. Through empirical experiments on various classification tasks, we demonstrate that our approach outperforms State-of-the-Art (SOTA) methods.
- We propose a goodness metric for the CG method that evaluates the balance between the average shift in class probability and the distance of the generated counterfactuals from the original input.
- We introduce a quantitative framework to explain model decisions by identifying associations between features that contribute to a specific model outcome.

## Getting Started

### Installing Dependencies

```sh
# Create a new environment
python3 -m venv .venv

# Activate a virtual environment
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Project Structure

All experiments were conducted separately for each dataset. The codebase is structured as follows:

#### Dataset-Specific Folders

- `adult_dataset/`: Contains all experimental results and evaluations associated with the Adult Dataset.
- `german_credit_risk/`: Contains all experimental results and evaluations associated with the German Credit Dataset.
- `pima_diabetes_dataset/`: Contains all experimental results and evaluations associated with the PIMA Indian Diabetes Dataset.
- `taiwanese_credit_dataset/`: Contains all experimental results and evaluations associated with the Taiwanese Credit Dataset.
- `grammars/`: Includes all the grammars used in the experiments.

#### Folder Structure Within Each Dataset

Each dataset folder contains the following files and subfolders:

- `evaluation/`: Contains all code and results related to the evaluation of the experiments.
- `model_training/`: Includes all code for dataset creation and model training.
- `output/`: Stores the generated counterfactuals and phenotypes.
- `Counterfactual_generation_multiprocess.py` & `optimization_algorithm.py`: Implements the GraCo method (multiprocess) for generating counterfactuals using both NSGA-II and NSGA-III.
- `Counterfactual_generation.py` & `optimization_algorithm.py`: Implements the GraCo method for generating counterfactuals using both NSGA-II and NSGA-III.
- `functions.py`: Contains functions utilized in the evolutionary process.
- `grape.py`: Contains the code for grammatical evolution.
- `Analyse_phenotype_globally.ipynb`: Provides global explainability using global phenotypes.
- `analyse_phenotype_graph.ipynb`: Provides local explainability for each input using the generated phenotypes.
- `feature_interaction_analysis/`: Contains phenotype analysis focused on feature interaction.
- `Phenotype_perturbation.ipynb`: Analyzes the impact of perturbing phenotypes on the model's outcomes.
- `PIMA_Hypervolume_analysis.ipynb`: Conducts hypervolume analysis.

## License

This project is licensed under the MIT License.

## Citation

If you use GraCo in your research, please cite our work:

```
@inproceedings{your_citation,
  author    = {Your Name and Co-authors},
  title     = {GraCo: Towards GRammar-Assisted Counterfactuals},
  booktitle = {GECCO 2025},
  year      = {2025}
}
```


