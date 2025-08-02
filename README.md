# Notes on DSP

A series of notes on Digital Signal Processing topics

---

## Project Overview

<!-- Add a brief overview of the project here. -->

---

## Installation Steps

Please read [install.md](docs/install.md) for details on how to set up this project.

---

## Key Features

- **Dependency Management**: Using Poetry for reliable dependency management
- **Data Version Control**: Optional DVC integration for data versioning
- **ML Experiment Tracking**: Optional MLflow integration for experiment tracking
- **Interactive Dashboards**: Optional Streamlit integration for creating dashboards
- **Jupyter Support**: Full support for Jupyter notebooks
- **Code Quality**: Pre-configured with black, ruff, and mypy
- **Testing**: pytest setup with coverage reporting
- **Documentation**: MkDocs setup with Material theme

### Quick Start

1. Install Poetry (see [Installation Guide](docs/install.md))
2. Clone the repository
3. Install dependencies:

   ```bash
   poetry install
   ```

4. Activate the environment:

   ```bash
   poetry env activate
   ```

---

## Notebooks Index

## Spectral Analysis

- [Sinc and Rectangular Functions](notebooks/spectral_analysis/sinc-pulse.ipynb): Exploration of the sinc function and its relationship with rectangular pulses in the frequency domain.
- [Zero-phase Zero-padding](notebooks/spectral_analysis/zero-phase-zero-padding.ipynb): Analysis of zero-padding in the frequency domain and its effects on spectral resolution.

### Analysis Windows

- [Rectangular Window](notebooks/analysis_windows/01-rectangular.ipynb): Analysis of the rectangular window function in DSP.
- [Triangular Window](notebooks/analysis_windows/02-triangular.ipynb): Analysis of the triangular window function in DSP.
- [Hann Window](notebooks/analysis_windows/03-hann.ipynb): Analysis of the Hann window function in DSP.
- [Hamming Window](notebooks/analysis_windows/04-hamming.ipynb): Analysis of the Hamming window function in DSP.
- [Blackman Window](notebooks/analysis_windows/05-blackman.ipynb): Analysis of the Blackman window function in DSP.
- [Blackman-Harris Window](notebooks/analysis_windows/06-blackman-harris.ipynb): Analysis of the Blackman-Harris window function in DSP.
- [Kaiser Window](notebooks/analysis_windows/07-kaiser.ipynb): Analysis of the Kaiser window function in DSP.

## Time-Frequency Analysis

- [Frequency Modulated Signals](notebooks/time-frequency_analysis/frequency-modulated-signals.ipynb): Exploration of frequency modulation and its effects on signal representation in the time-frequency domain.

---

## Basic Usage for Data Science Tasks

### Working in Notebooks

To enable autoreload in notebooks use the following magic command at the start of your notebook. This will automatically reload any changes to your code.

```python
%load_ext autoreload
%autoreload 2
```

### Data Management

```python
from dsp.utils.paths import data_raw_dir
from dsp.data.data_loader import load_csv

# Process raw data
df = load_csv(data_raw_dir('input.csv'))
```

### Model Development

```python
from dsp.utils.paths import data_processed_dir
from dsp.models import train_model

# Train model
model = train_model(
    data_path=data_processed_dir('train.csv'),
    model_type='random_forest'
)
```

### Visualization

```python
from dsp.utils.paths import reports_figures_dir
from dsp.visualization.visualize import plot_distribution

# Create visualization
plot_distribution(
    data=results_df,
    save_path=reports_figures_dir('results.png'),
)
```

### Local Experiment Tracking

When traicking experiments locally it is recommended to MLflow set up the tracking URI to the `models/` directory to store the artifacts and database files. To achieve this, you can use the following code snippet in your scripts or notebooks:

```python
import mlflow
from dsp.utils.paths import models_dir

# Define tracking and artifact paths
mlflow_db_path = models_dir("mlruns.db").as_posix()
mlflow_artifacts_path = models_dir("mlruns").as_posix()
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")

# Check if the experiment already exists and create it if not
experiment_name = "my_experiment"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(
        name=experiment_name,
        artifact_location=f"file:///{mlflow_artifacts_path}"
    )
mlflow.set_experiment(experiment_name)

print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
print(f"MLflow artifacts will be stored in: {mlflow_artifacts_path}")
```

Then you can use MLflow to log parameters, metrics, and artifacts as usual, for example:

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create dummy data and train a model
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)


# Log parameters and metrics
with mlflow.start_run():
    mlflow.log_param("fit_intercept", model.fit_intercept)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(
        model,
        name="model"
    )

print("Run logged successfully.")

```

MLflow still creates an empty `mlruns/` directory in the current working directory, but doesn't use it. This happens because:

- MLflow initializes its default file-based backend store (`FileStore`) by default pointing to `./mlruns`, even when you're using a database URI like `sqlite:///....`

- It's a side effect of how MLflow internally checks or sets up the default experiment (ID = 0) and stores backend metadata briefly, even if unused.

To avoid confusion, you can safely remove the `mlruns/` directory in your current working directory. All relevant data will be stored in the `models/` directory as specified. Yo can also run the following command to remove the `mlruns/` directory:

```python
import os
import shutil
cwd_mlruns = os.path.join(os.getcwd(), "mlruns")
if os.path.isdir(cwd_mlruns) and not os.listdir(cwd_mlruns):
    shutil.rmtree(cwd_mlruns)
```

Then Launch the MLflow UI to visualize your experiments:

```bash
mlflow ui --backend-store-uri sqlite:///<path_to_your_project>/models/mlruns.db 
```

---

## Project Organization

Please refer to the project structure tree in the [project_structure.md](docs/project_structure.md) for a detailed overview of the directory layout and file organization.

---

## Documentation

- [Installation Guide](docs/install.md): Detailed installation instructions
- [User Guide](docs/user_guide.md): How to use the project
- [Developer Guide](docs/developer_guide.md): Development guidelines
- [Project Structure](docs/project_structure.md): Detailed project structure
- [Contributing Guide](docs/contributing.md): How to contribute
- [Code of Conduct](docs/code_of_conduct.md): Community guidelines

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
