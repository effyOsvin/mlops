import fire

from mlops.infer import Infer
from mlops.train import Train


def train(
    path_to_data: str = "https://www.dropbox.com/s/gqdo90vhli893e0/data.zip",
    num_epoch: int = 10,
) -> None:
    Train().train_model(path_to_data, num_epoch)


def infer() -> None:
    Infer().make_infer()


if __name__ == "__main__":
    fire.Fire()
