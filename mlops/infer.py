import torch
from common.data_prep import test_data, valid_data
from common.runner import CNNRunner


class Infer:
    def make_infer(self):
        ckpt_name_cnn = "./mlops/bin/model_cnn.ckpt"
        best_model_cnn = None
        with open(ckpt_name_cnn, "rb") as f:
            best_model_cnn = torch.load(f)

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        opt = torch.optim.Adam(best_model_cnn.parameters(), lr=1e-3)

        runner_cnn = CNNRunner(best_model_cnn, opt, device, ckpt_name_cnn)
        val_batch_gen = valid_data()
        test_batch_gen = test_data()

        val_stats = runner_cnn.validate(val_batch_gen, phase_name="val")
        test_stats = runner_cnn.validate(test_batch_gen, phase_name="test")

        if val_stats["f1"] > 0.75 and test_stats["f1"] > 0.75:
            print("You have achieved the baseline for this task.")
        else:
            print("Train for some more time.")

        runner_cnn.predict(test_batch_gen)
