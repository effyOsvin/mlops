import fire
from infer import make_infer
from train import make_train


def train(
    path_to_data: str = "https://www.dropbox.com/s/gqdo90vhli893e0/data.zip",
    num_epoch: int = 10,
) -> None:
    make_train(path_to_data, num_epoch)


def infer() -> None:
    make_infer()


if __name__ == "__main__":
    fire.Fire()
