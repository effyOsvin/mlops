import os

from common.train_model import train_model


def load_data(path_to_data):
    # Training set with 11K images
    os.system(f"wget {path_to_data}")
    os.system("unzip -qq data.zip -d mlops/data")
    os.system("rm data.zip")


def make_train(path_to_data, num_epoch):
    ckpt_name_cnn = "./mlops/bin/model_cnn.ckpt"

    if not os.path.exists("./mlops/data"):
        load_data(path_to_data)

    train_model(ckpt_name_cnn, num_epoch)
