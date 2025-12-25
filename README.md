# Feature Selection Layer: A Feature Selection Approach for Multi-Class Classification on Neural Networks

> This project aims to create an extension to Feature Selection Layer (FSL) able to assign label-specific feature weights.

In the era of data, deep learning models are becoming increasingly influential in
uncovering patterns and making predictions that directly impact our society. One
of the primary challenges hindering their broader adoption in business and science is
their inherent lack of interpretability. Understanding and justifying how these mod-
els arrive at their decisions is crucial, especially in sensitive domains like healthcare,
biology, and finance. While neural networks excel at discovering hidden patterns
in massive, high-dimensional datasets, they can also inadvertently learn to exploit
biases in the data to achieve high accuracy, leading to biased predictions. This
bias is particularly dangerous when predictions depend on illegitimate factors that
can impact people’s lives. For example, models can mistakenly define gender as
a relevant factor for job performance or race as a factor for criminality, reinforc-
ing social inequalities. In recent years, various post-hoc methods have emerged to
enhance model interpretability by assigning weights to features based on their rel-
evance. These methods typically focus on identifying the most relevant features
for a given prediction, often considering the nuanced relationships between features
and different labels in multi-class problems. To leverage the inner workings of neu-
ral networks for this purpose, embedded methods have been developed to learn
feature importance jointly with the model during the training phase. These meth-
ods generally identify the most relevant features without considering the nuanced
relationships between features and different labels in multi-class problems, unlike
post-hoc techniques. Among these methods is the Feature Selection Layer, which
adds a new layer between the input and the neural network. This layer uses fea-
ture weights that directly affect predictions and are learned jointly with the rest
of the model’s parameters. However, like other embedded methods, the Feature
Selection Layer can only learn general feature weights. To address this limitation,
we propose an extension to Feature Selection Layer designed to capture the specific
relationships between features and individual labels. Our method was evaluated on
both synthetic and real-world datasets supported by multiple evaluation methods
and compared with state-of-the-art post-hoc techniques. It successfully identified
the most relevant features for different labels.

## Getting Started

You can execute this project using Docker for maximum reproducibility or locally if you need more control over your environment (e.g., specific CUDA versions).

### Execute from Docker (Recommended)

This method ensures all dependencies, including the required CUDA 13 environment, are set up correctly.

0.  For GPU usage the NVIDIA Container Toolkit it necessary
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    > Restart Docker
1.  Build the Docker image:
    docker build -t label-specific-fsl .
2.  Run the image:
    docker run --gpus all label-specific-fsl
    > Note: The Dockerfile is pre-configured to use CUDA 13. If you require a different version, local execution is recommended.

---

### Local Execution

This method requires a pre-existing Python environment and uv for dependency management.

#### Requirements & Setup

1.  Install uv (a fast Python package installer and virtual environment creator):
    curl -LsSf https://astral.sh/uv/install.sh | sh
2.  Create and activate the virtual environment:
    uv venv
    source .venv/bin/activate
3.  Install Dependencies:
    * Install core dependencies:
        uv pip install -r requirements.txt
    * Install PyTorch and CUDA 13 specific dependencies:
        uv pip install -r requirements-torch.txt --extra-index-url https://download.pytorch.org/whl/cu130
    > CUDA Version Note: This setup is optimized for CUDA 13. If you need CUDA 12 or another version, skip the requirements-torch.txt step and install the appropriate PyTorch version manually from the official PyTorch website.

---

## How to Execute Locally

Once the environment is set up, you need to configure the evaluation parameters before running the main script. All configuration files are located under the config/ directory.

### 1. Configure the Feature Weighting Algorithms

In the file run_src.evaluation.py, define the list of feature weighting algorithms (selectors) you want to compare in the selectors_types list.

* Example:
    selectors_types = [MFSLayerV1ReLUSelector, LassoSelectorWrapper, LIMESelectorWrapper, DeepSHAPSelectorWrapper]
    > All available methods are already imported in the script.

### 2. Setup General Execution Options

Review and adjust framework options and dataset paths in config/general_config.py. This controls the overall execution flow.

### 3. Setup Predictors

In config/predictor_types_config.py, specify the machine learning models (predictors) that will be used to evaluate the performance of the feature subsets selected by the algorithms.

* Current Options: SVC and Neural Network.

### 4. Setup Stability Metrics

Define the metrics for assessing the consistency of the feature selection process in config/stability_metrics_config.py.

* Current Options: Jaccard, Spearman, Pearson, and Kuncheva.

### 5. Execute the Analysis

With all configurations and dependencies prepared, run the main script:

python main.py

## WTSNE Error
`qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.`

Try:
`sudo apt-get install --reinstall libxcb-xinerama0 libxcb-xinerama0-dev libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0 libxcb-xkb1 libxcb-shape0 libxkbcommon-x11-0`