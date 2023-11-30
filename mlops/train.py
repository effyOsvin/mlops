from common.train_model import train_model


def main():
    ckpt_name_cnn = "./mlops/bin/model_cnn.ckpt"
    train_model(ckpt_name_cnn)


if __name__ == "__main__":
    main()
