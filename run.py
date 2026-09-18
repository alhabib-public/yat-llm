"""
Streamlit front end for training the character model and running the GP optimiser.

Launch with ``make run-streamlit-app``.
"""

import streamlit as st
from app.main import train_and_validate_llm_models, optimise_hyperparameters_via_gp
import matplotlib.pyplot as plt


def run_app():             
    """
    Render the sidebar controls and results.

    "Run Training" trains one model with the chosen hyperparameters and shows the
    training loss curve and generated text samples. "Optimise GP" runs the
    hyperparameter search and then retrains with the result.
    """
    # Sidebar for hyperparameters
    st.sidebar.header("Hyperparameters")
    strm_learning_rate = st.sidebar.number_input(
        "Learning Rate",
        value=0.0001,
        min_value=0.000000001,
        max_value=1.0,
        step=0.00001,
        format="%0.9f",
    )
    strm_depth = st.sidebar.number_input(
        "Depth", value=2, min_value=1, max_value=10, step=1
    )
    strm_width = st.sidebar.number_input(
        "Width", value=8, min_value=1, max_value=20, step=1
    )
    strm_n_embds = st.sidebar.number_input(
        "Embedding Length", value=48, min_value=1, max_value=10000000, step=1
    )
    strm_epochs = st.sidebar.number_input(
        "Training Iterations", value=5000, min_value=1, max_value=10000000, step=1
    )
    strm_block_size = st.sidebar.number_input(
        "Block Size", value=100, min_value=10, max_value=1000, step=1
    )

    if st.sidebar.button("Run Training"):
        with st.spinner("Training in progress..."):
            accuracies, all_completions = train_and_validate_llm_models(
                strm_learning_rate,
                strm_n_embds,
                strm_epochs,
                strm_block_size,
                strm_depth,
                strm_width,
            )

        # Plotting
        st.header("Training Loss Trend")
        plt.figure(figsize=(10, 5))
        plt.plot(accuracies, label="Training Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss Trend")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        for i, t in enumerate(all_completions):
            st.subheader(f"Iteration {i}:")
            st.text(t)

    st.sidebar.header("Optimiser")

    strm_optimisation_iterations = st.sidebar.number_input(
        "Optimisation Iterations", value=10, min_value=10, max_value=1000, step=1
    )

    if st.sidebar.button("Optimise GP"):
        with st.spinner("Optimisation in progress..."):
            result = optimise_hyperparameters_via_gp(strm_optimisation_iterations)
            accuracies, all_completions = train_and_validate_llm_models(*result)

        # Plotting
        st.header("Training Loss Trend")
        plt.figure(figsize=(10, 5))
        plt.plot(accuracies, label="Training Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss Trend")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        for i, t in enumerate(all_completions):
            st.subheader(f"Iteration {i}:")
            st.text(t)


if __name__ == "__main__":
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image("./LogoGreen_New.png", width=150)
    st.header("Character Auto-regression Optimiser/Ablation")

    run_app()
