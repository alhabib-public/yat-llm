"""
Training, validation and hyperparameter search for the feedforward character model.

Entry points:
    - ``train_and_validate_llm_models``: train one model for a given set of hyperparameters.
    - ``optimise_hyperparameters_via_gp``: Bayesian (Gaussian-process) search over
      the space defined in ``app.macros.optimisation_space``.
"""

import torch

from app.trainer import Trainer
from app.model import Feedforward
from app.utils import set_seed, setup_logging

from app.dataset import CharDataset
from app.macros import get_all_config, optimisation_space

from skopt import gp_minimize
from skopt.utils import use_named_args


def get_val_dataset(config, val_data):
    """
    Build every sliding-window example from the encoded validation text.

    Args:
        config: Full CfgNode; ``config.model.block_size`` sets the window length.
        val_data: List of token indices for the validation split.

    Returns:
        Tuple ``(inputs, targets)`` of LongTensors, each with shape
        ``(len(val_data) - block_size, block_size)``. ``targets`` is ``inputs``
        shifted one token to the right.
    """
    inputs, targets = [], []
    for idx in range(len(val_data) - config.model.block_size):
        inputs.append(val_data[idx : idx + config.model.block_size])
        targets.append(val_data[idx + 1 : idx + config.model.block_size + 1])
    return (torch.tensor(inputs), torch.tensor(targets))


# For debugging purposes if you use IPDB, resolves a multiprocessing issue:
# https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
# You can safely ignore this
__spec__ = (
    "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
)


def train(config, train_dataset):
    """
    Build a ``Feedforward`` model and train it on ``train_dataset``.

    Every 10 iterations the training loss is logged and appended to the loss
    history. Every 500 iterations (including iteration 0) the model generates
    500 characters, starting from the first ``block_size`` characters of
    ``data/context.txt``.

    Args:
        config: Full CfgNode. ``config.model.vocab_size`` and
            ``config.model.block_size`` are overwritten from the dataset.
        train_dataset: A ``CharDataset`` for the training split.

    Returns:
        Tuple ``(model, loss_history, all_completions)``: the trained model, the
        list of sampled training losses, and the list of generated text samples.
    """
    print(config)
    setup_logging(config, 0)
    set_seed(config.system.seed)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = Feedforward(config.model)

    all_completions = []
    loss_history = []

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
            )
            loss_history.append(trainer.loss.item())

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # construct the used context
                with open("data/context.txt", "r") as f:
                    context = f.read()[: config.model.block_size]

                # sample from the model...
                x = torch.tensor(
                    [train_dataset.stoi[s] for s in context], dtype=torch.long
                )[None, ...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = "".join([train_dataset.itos[int(i)] for i in y])

                print(completion)
                all_completions.append(completion)

            # revert model to training mode
            model.train()

    trainer.set_callback("on_batch_end", batch_end_callback)
    trainer.run()

    return model, loss_history, all_completions


def train_and_validate_llm_models(
    learning_rate, n_embds, epochs, block_size, depth, width
):
    """
    Train a model on ``data/input.txt`` with the given hyperparameters.

    The first ``split_ratio`` of the text is used for training and the rest for
    validation. Validation is skipped when ``block_size`` exceeds
    ``trainer.maximum_validation_block_size`` because it runs in a single pass.

    Args:
        learning_rate: AdamW learning rate.
        n_embds: Embedding dimension per character.
        epochs: Number of training iterations (batches), not full passes over the data.
        block_size: Context length in characters.
        depth: Number of hidden layers (see ``Feedforward.build_fnn``).
        width: log2 of the first hidden layer's size.

    Returns:
        Tuple ``(loss_history, all_completions)``. See ``train``.
    """
    # construct the entire dataset
    with open("data/input.txt", "r") as f:
        data = f.read()

    config = get_all_config()
    config.model.learning_rate = learning_rate
    config.model.n_embd = n_embds
    config.model.block_size = block_size
    config.model.depth = depth
    config.model.width = width
    config.trainer.max_iters = epochs
    config.trainer.learning_rate = learning_rate

    split_idx = int(len(data) * config.trainer.split_ratio)
    train_dataset = CharDataset(config.model, data[:split_idx])

    model, loss_history, all_completions = train(config, train_dataset)

    validation_is_too_slow = block_size > config.trainer.maximum_validation_block_size
    if not validation_is_too_slow:
        # The validation dataset can be evaluated all at once
        val_data = [train_dataset.stoi[x] for x in data[split_idx:]]
        val_dataset = get_val_dataset(config, val_data)

        final_accuracy = model(val_dataset[0], val_dataset[1])[1] / len(val_dataset[0])
        print(f"Final accuracy of best model: {final_accuracy.tolist()}")

    # TODO: ideally, final_accuracy is used for optimisation, but due to slow runtimes
    # we are using training loss_history as an approximator of final_accuracy
    return loss_history, all_completions


def optimise_hyperparameters_via_gp(n_calls=10):
    """
    Search for good hyperparameters with Gaussian-process Bayesian optimisation.

    Each evaluation trains a full model via ``train_and_validate_llm_models`` and
    uses the final sampled training loss as the objective to minimise.

    Args:
        n_calls: Number of models to train during the search.

    Returns:
        ``scipy.optimize.OptimizeResult`` from ``skopt.gp_minimize``; the best
        hyperparameters are in ``result.x`` (in ``optimisation_space`` order) and
        the best loss is in ``result.fun``.
    """
    # Define the objective function
    space = optimisation_space()

    @use_named_args(space)
    def objective(learning_rate, n_embds, epochs, block_size, depth, width):
        loss_history, _ = train_and_validate_llm_models(
            learning_rate, n_embds, epochs, block_size, depth, width
        )
        return float(loss_history[-1])

    result = gp_minimize(objective, space, n_calls=n_calls, random_state=0, n_jobs=1)

    # Best hyperparameter values found
    print("Best parameters: {}".format(result.x))
    print("Best loss: {}".format(result.fun))

    return result


if __name__ == "__main__":
    optimise_hyperparameters_via_gp()
