# Test Case Descriptor Summarization 
This repository contains a multi-model test case summarization pipeline developed for the IATS/Motorola scholarship project. 
The pipeline processes test case descriptions from “The Test-Case Dataset” to generate concise summaries, addressing issues with incoherent outputs (e.g., “ambasadEasily”). 
It evaluates four transformer-based models using the ROUGE metric, supporting automated test case documentation for software testing workflows. ## Project Overview 

The project implements a summarization pipeline that: 
- **Cleans Data** Normalizes Title and Description fields, combining them into a Combined Input for richer context. 
- **Summarizes Test Cases** Uses four models:
  1. **Custom**: Fine-tuned `t5-small` on CNN/DailyMail dataset.
  2. **T5**: `t5-base` for robust English summarization.
  3. **Flan-T5**: `google/flan-t5-large` for superior zero-shot performance.
  4. **BART**: `facebook/bart-large-cnn` for coherent, state-of-the-art summarization. 
- **Evaluates Performance** Computes ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores, using Cleaned Title as the reference.
- **Outputs Results** - `generated_summaries.csv` with summaries and per-row ROUGE-1 scores
- A summary table of overall ROUGE scores in the Jupyter notebook The pipeline is optimized for a GTX 1070Ti GPU, with mixed precision recommended for `flan-t5-large` (~1.5 GB VRAM).

## Installation ### Prerequisites - Python 3.8+ - NVIDIA GPU (e.g., GTX 1070Ti) with CUDA support (optional for CPU-only execution) - Git LFS (if tracking large model files) 

### Setup 
1. **Clone the Repository** ```bash git clone https://github.com/Odalisio-Neto/Test-Case-Descriptor.git cd Test-Case-Descriptor ```
2. **Install Dependencies** ```bash conda env create -f env.yml conda activate test-case-descriptor ```
3. **Download Fine-Tuned Model** The Custom `t5-small` model is stored in `./summarization_model/final`. If not included, re-run fine-tuning: ```bash python model/train.py ```
4. **Verify Git LFS** (if used) ```bash git lfs install ```

## Usage 
1. **Prepare Data** Ensure `Test_cases.csv` is in the root directory with columns `Test Id`, `Title`, and `Description`.
2. **Run the Pipeline** - In Jupyter: ```bash jupyter notebook main.ipynb ``` - As a script: ```bash python main.py ```
3. **Output** - `generated_summaries.csv` with summaries and ROUGE-1 scores - Summary table in `main.ipynb`

## Evaluation Results
The models were evaluated using ROUGE metrics, with Cleaned Title as the reference summary. Below are the mean ± standard deviation scores: 
| Model | ROUGE‑1 | ROUGE‑2 | ROUGE‑L | ROUGE‑Lsum | 
|----------|-------------------|-------------------|-------------------|-------------------| 
| Custom | 0.7348 ± 0.3009 | 0.6746 ± 0.3696 | 0.7231 ± 0.3149 | 0.7231 ± 0.3149 | 
| T5 | 0.7094 ± 0.3191 | 0.6443 ± 0.3800 | 0.6948 ± 0.3322 | 0.6948 ± 0.3322 | 
| Flan‑T5 | 0.8442 ± 0.2572 | 0.8071 ± 0.3080 | 0.8417 ± 0.2613 | 0.8417 ± 0.2613 | 
| BART | 0.7805 ± 0.2793 | 0.7388 ± 0.3366 | 0.7758 ± 0.2849 | 0.7758 ± 0.2849 | 

## Insights 
- **Flan‑T5** leads with the highest scores (ROUGE‑1: 0.8442), excelling in precision and consistency.
- **BART** follows closely (ROUGE‑1: 0.7805), offering fluent and coherent summaries.
- **Custom** (fine‑tuned `t5-small`) outperforms T5 (0.7348 vs. 0.7094), showing promise for resource-constrained environments.
- **T5** lags due to its general-purpose training without task-specific fine‑tuning.
## Project Structure 
``` Test-Case-Descriptor/ 
  ├── main.py # Main script for summarization pipeline
  ├── main.ipynb # Jupyter notebook with pipeline and evaluation
  ├── model/
  │ ├── __init__.py # Model package initialization
  │ └── train.py # Script for fine‑tuning Custom t5-small
  ├── summarization_model/
  │ └── final/ # Fine‑tuned t5-small model
  ├── Test_cases.csv # Input dataset
  ├── generated_summaries.csv # Output summaries and ROUGE scores
  ├── env.yml # Conda environment configuration
  ├── .gitignore # Git ignore rules
  └── README.md # This file
```

## Future Work 
- **Domain-Specific Fine‑Tuning**: Enhance the Custom model with a test case‑specific dataset.
- **CLI Interface**: Develop a command‑line tool for easier pipeline execution.
- **CI/CD Integration**: Add GitHub Actions for automated testing and validation.
- **Model Optimization**: Implement quantization for `flan-t5-large` to reduce VRAM usage.
  
## License
  This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
## Contact
  For questions, contact **Odalisio Neto** or open an issue on GitHub.
