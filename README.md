# 🩸 Blood Donors Prediction

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
[![Project CI](https://github.com/erikdeluca/Blood-Donors/actions/workflows/ci.yml/badge.svg)](https://github.com/erikdeluca/Blood-Donors/actions/workflows/ci.yml)
[![Quarto Publish](https://github.com/erikdeluca/Blood-Donors/actions/workflows/publish.yml/badge.svg)](https://github.com/erikdeluca/Blood-Donors/actions/workflows/publish.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-4.3-blue?logo=r&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

This project aims to predict the number of donations made by a blood donor in the upcoming year based on previous donation history and the demographic informations available, as age and sex. The results are available in the website rendered by this repository: [Blood Donors Prediction](https://www.erikdeluca.it/Blood-Donors/).

The model built is an upgrade of an Hidden Markov Model. In the emission probabilities, a generalized linear model is used to take into account the demographic features of the donors.
In the transition probabilities and initial state probabilities, the covariates are managed in a Bayesian framework.
This setting allows to take into account the heterogeneity of the donors, improving the prediction performance and allowinig a better interpretability of the results.

![Model structure](slides/images/diagram_HMM-GLM.png)

### Results

The model perform better than a plain vanilla Generalized Linear Model. Moreover, the Hidden Markov Model structure allows to cluster the donors in dynamical groups, each one with its own characteristics. In the pictures below, the inferred hidden states are shown, along with the transition probabilities between them and in the last year the predicted donations for each donor.

![Prediction examples](img/prediction_examples_grid.png)

## Project Structure

This project hosts my master thesis work and an exam project for the course "Probabilistic Machine Learning" at the University of Trieste.
The repository is structured as follows:
- `app/`: contains the code for a Streamlit application in production on a Raspberry PI and accessible via [this link](https://blood-donors.erikdeluca.it/) 
- `app-quarto/`: contains the code for a web application to interactively explore the model results. It works with Quarto and Shiny for  Python
- `bibliography/`: contains the bibliography files used in the thesis write-up
- `data/`: contains the dataset used for the analysis
- `docs/`: contains the website generated with Quarto
- `img/`: contains images used in the website and for another purpose
- `models/`: contains the different models to avoid re-computation
- `notebooks/`: contains Jupyter notebooks and Quarto markdowns for data exploration, model development, and evaluation
- `python/`: contains Python scripts for data processing and model implementation, including adaptations of existing libraries
- `R/`: contains R scripts for data exploration and visualization. Furthermore, it contains some data in RDS format
- `slides/`: contains the slides used for the presentation of the thesis and the presentation of the exam project
- `tests/`: contains unit tests for the first page of the thesis
- `thesis/`: contains the Quarto markdown files for the master thesis write-up

## Installation

To run the project you need to have R, Python and Quarto installed on your machine.

### Quarto

You can find the installation instructions for Quarto [here](https://quarto.org/docs/get-started/).

### Python

To install the Python dependencies and replicate the environment used for the project, you can use `conda` with the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate blood-donors-prediction
```

### R

To install the R dependencies, you can use the `renv` package. First, install `renv` if you don't have it already:

```R
install.packages("renv")
```

Then, in the R console, run:

```R
renv::restore()
```

This will install all the required packages specified in the `renv.lock` file.

## Usage

To build the website, navigate to the root directory of the project and run:

```bash
quarto render
```

This will generate the website in the `docs/` folder.

To run the Shiny application, navigate to the `app/` directory and run the following command in the terminal:

```bash
quarto preview dashboard.qmd
```

### Streamlit app 

To run the Streamlit application in a container (simulating the production environment), install docker first (for linux systems):

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

Then run the app:

```bash
cd app
docker compose up --build
```

The app will be available at http://localhost:8501