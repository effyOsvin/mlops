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
python mlops/train.py
```

For infer a model, run:

```
python mlops/infer.py
```

For train and infer a resulting model, run:

```
python commands.py
```
