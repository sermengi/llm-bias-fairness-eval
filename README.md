# Bias and Fairness Evaluation in Large Language Models (LLMs)

### Overview

This repository is dedicated to evaluating Large Language Models (LLMs) with a focus on bias and fairness. The primary goal is to explore and implement methodologies from existing literature, reproduce their experiments in code, and compare results across different open-source LLMs.

By doing so, this project aims to provide an empirical foundation for assessing model behavior in contexts where fairness and impartiality are crucial.

### Motivation
As LLMs become integral to a wide range of applications—from content generation to decision support systems—their ethical implications must be carefully examined. Biased outputs can lead to unfair or harmful consequences, especially in sensitive scenarios like hiring decisions, loan approvals, or legal assessments.

This project is motivated by the need to:

- Fairly evaluate LLMs for potential biases.

- Promote transparency and accountability in model deployment.

- Support ethical and responsible AI usage in real-world applications.

### Project Structure
```bash
llm-bias-fairness-eval/
├── LICENSE
├── README.md
├── config.yaml
├── logs
│   └── logging.log
├── main.py
├── notebooks
│   ├── 01_literature_review.ipynb
│   ├── 02_data_exploration.ipynb
│   └── 03_general_research.ipynb
├── pyproject.toml
├── src
│   ├── __init__.py
│   ├── common.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── metrics.py
│   ├── models.py
│   ├── qualitative.py
│   ├── streamlit_app.py
│   └── visualization.py
├── tests
│   ├── __init__.py
│   └── test_metrics.py
└── uv.lock
```

### Installation
This project uses uv as the package and environment manager, which provides a fast and reproducible setup using pyproject.toml and uv.lock.

* Python 3.11 is required.
    > ℹ️ uv will automatically manage the virtual environment using the compatible Python version. However, since this project utilizes pytorch/xla, which currently supports up to Python 3.11, it is recommended to have Python 3.11 installed on your system.


1. **Clone the Repository**
    ```bash
    git clone https://github.com/sermengi/llm-bias-fairness-eval.git
    cd llm-bias-fairness-eval
    ```
2. **Install <code>uv</code>**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
3. **Create the Virtual Environment and Install Dependencies**
    ```bash
    uv sync
    ```
    This will create a virtual environment and install all dependencies as specified in <code>pyproject.toml</code> and locked in <code>uv.lock</code>.
4. **Set Up Pre-commit Hooks** <br>
    To ensure code quality and consistency, we use ruff for linting and pre-commit for formatting checks. <br>
    Set up the pre-commit hooks:
    ```bash
    pre-commit install
    ```
    Now, each commit will automatically trigger linting and formatting checks. For optionally, you can run against all the files
    ```bash
    pre-commit run --all-files
    ```
