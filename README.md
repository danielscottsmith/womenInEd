# Replication Package
>#### Women, World Society, and the Expansion of Academic Knowledge in Education.

This is a working project directory and will be made into a stable reproduction package, _eventually_. The main goal is to understand the evolution of research on gender within the field of education using every article published and hosted by JSTOR, 1900–2022. Currently, the strategy is to run topic models to identify the latent topic of gender/sex/sexuality, and then explain variation in article's focus on that topic over time using logitudinal regression model.

---
## To Do

1. write out rep steps 
1. consider inferential modeling
1. put data on dataverse
1. write example command lines

## Project Organization

Below is a map of the directory with annotations for general orientation.

---

    ├── LICENSE
    ├── README.md          <- The top-level README for researchers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump from JSTOR. 2 jsonl files: part 1 = 100K articles; part 2 ~32K articles.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for order of analysis).
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Script to read in and structure jsonl files as a dataframe.
        │   └── make_dataset.py
        │
        ├── features       <- Script to turn interim data into bag of words, dictionary, and corpus objects.
        │   └── build_features.py
        │
        ├── models         <- Scripts to train & evaluate MALLET models.
        │   ├── predict_model.py
        │   ├── eval_mallet.py
        │   └── train_mallet.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py


--------

<p><small>Project structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
