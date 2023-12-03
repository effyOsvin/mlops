import os

from common.train_model import train_model


class Train:
    def load_data(self, path_to_data):
        # Training set with 11K images
        os.system(f"wget {path_to_data}")
        os.system("unzip -qq data.zip -d mlops/data")
        os.system("rm data.zip")

    def train_model(self, path_to_data, num_epoch):
        ckpt_name_cnn = "./mlops/bin/model_cnn.ckpt"

        if not os.path.exists("./mlops/data"):
            self.load_data(path_to_data)

        train_model(ckpt_name_cnn, num_epoch)
