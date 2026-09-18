"""
Default configuration for the model, trainer and hyperparameter search.
"""

from app.utils import CfgNode as CN
from skopt.space import Real, Integer


def get_model_config():
    """
    Return the default model config.

    ``n_embd`` is set from the chosen hyperparameters; ``vocab_size`` and
    ``block_size`` are filled in from the dataset in ``app.main.train``.
    """
    C = CN()
    C.model_type = "feedforward"
    C.data_type = "chars"
    C.n_embd = None
    C.vocab_size = None
    C.block_size = None

    return C


def get_trainer_config():
    """
    Return the default trainer config.

    Includes device selection, DataLoader workers, AdamW settings, gradient
    clipping, the train/validation split ratio, and the largest ``block_size``
    for which validation is run.
    """
    C = CN()
    C.device = "auto"

    # dataloder parameters
    C.num_workers = 4

    # optimizer parameters
    C.max_iters = 1000
    C.batch_size = 64
    C.learning_rate = 5e-4
    C.betas = (0.9, 0.95)
    C.weight_decay = 0.1
    C.grad_norm_clip = 1.0
    C.split_ratio = 0.7
    C.maximum_validation_block_size = 200

    return C


def get_all_config():
    """Return the full config with ``system``, ``model`` and ``trainer`` sections."""
    C = CN()
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = "./out/chargpt"

    # model
    C.model = get_model_config()

    # trainer
    C.trainer = get_trainer_config()

    return C


def optimisation_space():
    """
    Return the hyperparameter search space for ``skopt.gp_minimize``.

    The order of dimensions matches the argument order of
    ``app.main.train_and_validate_llm_models``.
    """
    return [
        Real(1e-5, 1e-2, name="learning_rate"),
        Integer(48, 128, name="n_embds"),
        Integer(500, 5000, name="epochs"),
        Integer(10, 100, name="block_size"),
        Integer(1, 5, name="depth"),
        Integer(6, 12, name="width"),
    ]
