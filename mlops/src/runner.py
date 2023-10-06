from collections import defaultdict
import time
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, accuracy_score



class Runner:
    def __init__(self, model, opt, device, checkpoint_name=None):
        self._phase_name = None
        self.model = model
        self.opt = opt
        self.device = device
        self.checkpoint_name = checkpoint_name

        self.epoch = 0
        self.output = None
        self.metrics = None
        self._global_step = 0
        self._set_events()
        self._top_val_accuracy = -1
        self.log_dict = {
            "train": [],
            "val": [],
            "test": []
        }

    def _set_events(self):
        self._phase_name = ''
        self.events = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list)
        }

    def train(self, train_loader, val_loader, n_epochs, model=None, opt=None, **kwargs):
        self.opt = (opt or self.opt)
        self.model = (model or self.model)

        for _epoch in range(n_epochs):
            start_time = time.time()
            self.epoch += 1
            print(f"epoch {self.epoch:3d}/{n_epochs:3d} started")

            # training part
            self._set_events()
            self._phase_name = 'train'
            self._run_epoch(train_loader, train_phase=True)

            print(f"epoch {self.epoch:3d}/{n_epochs:3d} took {time.time() - start_time:.2f}s")

            # validation part
            self._phase_name = 'val'
            self.validate(val_loader, **kwargs)
            self.save_checkpoint()

    def _run_epoch(self, loader, train_phase=True, output_log=False):
        self.model.train(train_phase)

        _phase_description = 'Training' if train_phase else 'Evaluation'
        for batch in tqdm(loader, desc=_phase_description, leave=False):

            # forward pass through the model using preset device
            self._run_batch(batch)

            # train on batch: compute loss and gradients
            with torch.set_grad_enabled(train_phase):
                loss = self.run_criterion(batch)

            # compute backward pass if training phase
            # reminder: don't forget the optimizer step and zeroing the grads
            if train_phase:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        self.log_dict[self._phase_name].append(np.mean(self.events[self._phase_name]['loss']))

        if output_log:
            self.output_log()

    def save_checkpoint(self):
        val_accuracy = self.metrics['accuracy']
        # save checkpoint of the best model to disk
        if val_accuracy > self._top_val_accuracy and self.checkpoint_name is not None:
            self._top_val_accuracy = val_accuracy
            with open(self.checkpoint_name, 'wb') as checkpoint_file:
                torch.save(self.model, checkpoint_file)

    @torch.no_grad()
    def validate(self, loader, phase_name='val', **kwargs):
        self._phase_name = phase_name
        self._reset_events(phase_name)
        self._run_epoch(loader, train_phase=False, output_log=True)
        return self.metrics

    def _run_batch(self, batch):
        X_batch, y_batch = batch

        # update the global step in iterations over source data
        self._global_step += len(y_batch)

        # move data to target device
        X_batch = X_batch.to(self.device)

        # run the batch through the model
        self.output = self.forward(X_batch)

    def run_criterion(self, batch):
        raise NotImplementedError("To be implemented")

    def output_log(self):
        raise NotImplementedError("To be implemented")

    def _reset_events(self, event_name):
        self.events[event_name] = defaultdict(list)

    def forward(self, img_batch):
        logits = self.model(img_batch)
        output = {
            "logits": logits,
        }
        return output


class CNNRunner(Runner):
    def __init__(self, model, opt, device, checkpoint_name=None):
        super().__init__(model, opt, device, checkpoint_name)

    def run_criterion(self, batch):
        X_batch, label_batch = batch
        label_batch = label_batch.to(self.device)

        logit_batch = self.output['logits']

        loss = F.cross_entropy(logit_batch, label_batch)

        scores = F.softmax(logit_batch, 1).detach().cpu().numpy()[:, 1].tolist()
        labels = label_batch.detach().cpu().numpy().ravel().tolist()

        # log some info
        self.events[self._phase_name]['loss'].append(loss.detach().cpu().numpy())
        self.events[self._phase_name]['scores'].extend(scores)
        self.events[self._phase_name]['labels'].extend(labels)

        return loss

    def save_checkpoint(self):
        val_accuracy = self.metrics['accuracy']
        # save checkpoint of the best model to disk
        if val_accuracy > self._top_val_accuracy and self.checkpoint_name is not None:
            self._top_val_accuracy = val_accuracy
            with open(self.checkpoint_name, 'wb') as checkpoint_file:
                torch.save(self.model, checkpoint_file)

    def output_log(self, **kwargs):
        scores = np.array(self.events[self._phase_name]['scores'])
        labels = np.array(self.events[self._phase_name]['labels'])

        assert len(labels) > 0, print('Label list is empty')
        assert len(scores) > 0, print('Score list is empty')
        assert len(labels) == len(scores), print('Label and score lists are of different size')

        self.metrics = {
            "loss": np.mean(self.events[self._phase_name]['loss']),
            "accuracy": accuracy_score(labels, np.int32(scores > 0.5)),
            "f1": f1_score(labels, np.int32(scores > 0.5))
        }
        print(f'{self._phase_name}: ', end='')
        print(' | '.join([f'{k}: {v:.4f}' for k, v in self.metrics.items()]))

        self.save_checkpoint()
