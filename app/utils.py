"""
Shared helpers: seeding, run logging and the ``CfgNode`` config container.

You shouldn't need to make any changes to this file.
"""

import os
import sys
import random
from ast import literal_eval

import numpy as np
import torch


def set_seed(seed):
    """Seed Python, NumPy and PyTorch (CPU and all GPUs) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(config, run_idx):
    """
    monotonous bookkeeping

    Creates ``config.system.work_dir`` if needed and writes the command-line
    arguments of the current process to ``args.txt`` inside it.

    Args:
        config: Full CfgNode (see ``app.macros.get_all_config``).
        run_idx: Currently unused.
    """
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, "args.txt"), "w") as f:
        f.write(" ".join(sys.argv))


class CfgNode:
    """
    a lightweight configuration class inspired by yacs

    Attributes are set freely and nodes can be nested, e.g.
    ``C.model = CfgNode(); C.model.n_embd = 48``.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """need to have a helper to support nested indentation for pretty printing"""
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [" " * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """return a dict representation of the config"""
        return {
            k: v.to_dict() if isinstance(v, CfgNode) else v
            for k, v in self.__dict__.items()
        }

    def merge_from_dict(self, d):
        """Shallow-update the config's attributes from dict ``d``."""
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split("=")
            assert len(keyval) == 2, (
                "expecting each override arg to be of form --arg=value, got %s" % arg
            )
            key, val = keyval  # unpack

            # first translate val into a python object
            try:
                # need some explanation here.
                # - if val is simply a string, literal_eval will throw a ValueError
                # - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                val = literal_eval(val)
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == "--"
            key = key[2:]  # strip the '--'
            keys = key.split(".")
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(
                obj, leaf_key
            ), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)
