"""
Custom feedforward model for character-level next-token prediction.

Unlike a transformer, the model sees its whole context window at once: the
embeddings of all ``block_size`` input tokens are flattened into one vector and
passed through a stack of fully-connected layers that predicts the next token.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from app.utils import CfgNode as CN


def relu_activation(x):
    """Apply the ReLU non-linearity used between the hidden layers."""
    return F.relu(x)


class Feedforward(nn.Module):
    """
    A fully-connected neural network that consumes block_size characters to produce the next one.

    Architecture::

        Embedding(vocab_size, n_embd)            # (b, t) -> (b, t, n_embd)
        flatten                                  # -> (b, t * n_embd)
        Linear -> ReLU -> ... -> Linear -> ReLU  # hidden widths 2**width, halving each layer
        Linear(no bias)                          # -> (b, vocab_size) logits

    Only a single next-token prediction is made per input sequence, so the
    output always has a time dimension of 1.
    """

    @staticmethod
    def get_default_config():
        """
        Return a CfgNode with the model's default configuration.

        ``vocab_size`` and ``block_size`` must be filled in from the dataset before
        the model is built, and ``n_embd``, ``width`` and ``depth`` must also be
        set (see ``app.main.train_and_validate_llm_models``).
        """
        C = CN()
        C.model_type = "feedforward"
        C.n_embd = None
        C.hidden_dim = None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        return C

    def __init__(self, config):
        """
        Build the token embedding table and the fully-connected layers.

        Args:
            config: Model CfgNode providing ``vocab_size``, ``block_size``,
                ``n_embd``, ``width`` and ``depth``.
        """
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.block_size = config.block_size
        self.build_fnn(config)

    def build_fnn(self, config):
        """
        Here we define the FNN model using config.width and config.depth as the parameters
        controlling the capacity and non-linearity of the model, respectively.

        The first layer maps the flattened embeddings (``block_size * n_embd``) to
        ``2**width`` units. Each of the ``depth`` hidden layers then halves the
        number of units, and a final bias-free layer projects to ``vocab_size``
        logits. ``width`` is raised to at least ``depth + 1`` so the smallest
        hidden layer always has 2 or more units.

        Args:
            config: Model CfgNode providing ``width``, ``depth``, ``block_size``,
                ``n_embd`` and ``vocab_size``.
        """
        width, depth = max(config.width, config.depth + 1), config.depth
        self.fnn_layers = nn.ModuleList(
            [nn.Linear(config.block_size * config.n_embd, 2**width)]
        )
        for d in range(depth):
            self.fnn_layers.append(nn.Linear(2 ** (width - d), 2 ** (width - d - 1)))
        self.fnn_layers.append(
            nn.Linear(2 ** (width - depth), config.vocab_size, bias=False)
        )

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases).
        We are then returning the PyTorch optimizer object.

        Linear weights are decayed; biases and embedding weights are not.

        Note: training currently goes through ``Trainer.configure_optimizers``,
        which duplicates this logic, so this method is not called by ``app.main``.

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

        for mn, m in self.named_modules():
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
        param_dict = {pn: p for pn, p in self.named_parameters()}
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

    def forward(self, idx, targets=None):
        """
        Predict logits for the token that follows each input sequence.

        Args:
            idx: LongTensor of token indices, shape ``(b, t)``. ``t`` must be at
                most ``block_size``. Since the first layer expects exactly
                ``block_size * n_embd`` inputs, in practice ``t == block_size``.
            targets: Unused. Accepted for API compatibility with the trainer; the
                loss is computed in ``Trainer.run``.

        Returns:
            Tensor of logits with shape ``(b, 1, vocab_size)``.
        """
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        tok_emb = self.wte(idx)
        # Flatten embeddings in preparation of the fully-connected layers
        x = tok_emb.reshape(b, -1)

        for i, layer in enumerate(self.fnn_layers):
            x = layer(x)
            before_last_layer = i < (len(self.fnn_layers) - 1)
            if before_last_layer:
                x = relu_activation(x)

        # Reshape output to be consistent with the rest of the training framework
        return x.reshape(b, 1, -1)

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Args:
            idx: LongTensor of shape ``(b, t)`` holding the conditioning tokens.
                Only the last ``block_size`` tokens are fed to the model at each step.
            max_new_tokens: Number of tokens to append.
            temperature: Logits are divided by this before the softmax; values
                below 1.0 sharpen the distribution, above 1.0 flatten it.
            do_sample: If True, sample from the distribution; otherwise take the
                most likely token (greedy decoding).
            top_k: If set, only the ``top_k`` most likely tokens can be chosen.

        Returns:
            LongTensor of shape ``(b, t + max_new_tokens)``: the input followed by
            the generated tokens.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
