# mlops
## Task

We solve the problem of image classification using a dataset
[Dogs vs Cats](https://www.dropbox.com/s/gqdo90vhli893e0/data.zip)
.

## Usage
### Setup

```
poetry install
pre-commit install
pre-commit run -a
```

### Run experiments

For train the model and save it afterward, run:

```
poetry run python mlops/commands.py train
```

For infer a model, run:

```
poetry run python mlops/commands.py infer
```
