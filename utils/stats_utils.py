from transformers import PreTrainedModel
import torch.nn as nn
import pandas as pd
import torch


def count_parameters(model):
    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        # Count total parameters
        total_params += param.numel()

        # Count trainable parameters
        if param.requires_grad:
            trainable_params += param.numel()

    return trainable_params, total_params


def model_memory_used(model: PreTrainedModel):
    # Number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Memory usage
    # Assuming float32 parameters, each parameter is 4 bytes
    memory_bytes = num_params * 4
    memory_gigabytes = memory_bytes / (1024**3)

    return {"num_params": num_params, "memory_gb": memory_gigabytes}




# model is trained
def print_some_model_weights(model, target_modules: list = ["embed_tokens"]):
    import matplotlib.pyplot as plt

    for name, param in model.named_parameters():
        if any([x in name for x in target_modules]) and "lora" in name.lower():
            print(f"Best model weights {name} {param.shape}:")
            print(param)
            param_numpy = param.detach().cpu().to(torch.float32).numpy()[:2, :10]
            plt.imshow(param_numpy / param_numpy.max())
            plt.title(name)
            plt.show()
            break


def plot_stats(test_df: pd.DataFrame, run, target_col: str):
    target_col = "RET_10D_pos"

    assert (test_df.labels == test_df[target_col].values).sum() == len(test_df)

    # confusion matrix between preds and labels
    true_pos = (test_df.preds == 1) & (test_df[target_col].values == 1)
    true_neg = (test_df.preds == 0) & (test_df[target_col].values == 0)
    false_pos = (test_df.preds == 1) & (test_df[target_col].values == 0)
    false_neg = (test_df.preds == 0) & (test_df[target_col].values == 1)

    print(f"true_pos: {true_pos.sum()}")
    print(f"true_neg: {true_neg.sum()}")
    print(f"false_pos: {false_pos.sum()}")
    print(f"false_neg: {false_neg.sum()}")

    z = [
        [true_neg.sum(), false_neg.sum()],
        [false_pos.sum(), true_pos.sum()],
    ]

    import plotly.express as px

    fig = px.imshow(z, labels=dict(x="True", y="Predicted", color="Count"))
    run["test/confusion_matrix"] = fig
    fig = px.histogram(
        test_df,
        x="preds",
        title="Histogram of predictions",
    )
    run["test/histogram_preds"] = fig
    # for test labels
    fig = px.histogram(
        test_df,
        x=target_col,
        title="Histogram of test labels",
    )
    run["test/histogram_labels"] = fig


