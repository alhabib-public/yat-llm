"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network.
You shouldn't need to make any changes to this file.
"""

import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

class Trainer:
    """
    Runs the optimisation loop for a model on a dataset.

    Batches are drawn with replacement from ``train_dataset`` until
    ``config.max_iters`` iterations have run. After every iteration the
    ``"on_batch_end"`` callbacks are invoked with the trainer, which exposes
    ``iter_num``, ``iter_dt`` (seconds for the last iteration) and ``loss``.
    """

    def __init__(self, config, model, train_dataset):
        """
        Args:
            config: Trainer CfgNode (see ``app.macros.get_trainer_config``).
                ``device="auto"`` selects CUDA when available, else CPU.
            model: The ``nn.Module`` to train; it is moved to the chosen device.
            train_dataset: A ``torch.utils.data.Dataset`` yielding ``(x, y)`` pairs.
        """
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.default_loss_fn = nn.CrossEntropyLoss()

        # determine the device we'll train on
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        """Register ``callback(trainer)`` to run on ``onevent``, keeping existing ones."""
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        """Register ``callback(trainer)`` as the only callback for ``onevent``."""
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        """Call every callback registered for ``onevent`` with this trainer."""
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        """
        Train the model for ``config.max_iters`` iterations.

        Each iteration takes one batch, computes the cross-entropy loss of the
        model's single next-token prediction against the last target token
        (``y[:, -1]``), clips gradients to ``config.grad_norm_clip`` and takes one
        AdamW step. The latest loss is stored on ``self.loss``.
        """
        model, config = self.model, self.config

        # setup the optimizer
        # self.optimizer = model.configure_optimizers(config)
        self.optimizer = self.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset, replacement=True, num_samples=int(1e10)
            ),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()

        data_iter = iter(train_loader)

        # TODO: streamline robust data iterations
        # sometimes this step breaks due to multi_proc issues
        # on small CPUs, try it until it succeeds to be safe
        while True:
            try:
                data_iter = iter(train_loader)
                break
            except:
                pass

        # run all batches through training
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            logits = model(x, y)

            targets = y
            if targets is not None:
                # The fully connected layer predicts only the last character
                targets = targets[:, -1]
                self.loss = self.default_loss_fn(logits.squeeze(), targets)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks("on_batch_end")

            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases).
        We are then returning the PyTorch optimizer object.

        Linear weights are decayed; biases and embedding weights are not.

        Args:
            train_config: Trainer CfgNode providing ``weight_decay``,
                ``learning_rate`` and ``betas``.

        Returns:
            torch.optim.AdamW: Optimizer with separate decay / no-decay groups.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.Embedding,)

        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer
