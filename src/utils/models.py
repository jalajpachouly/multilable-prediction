"""Model building functions."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D
from .config import MLPConfig, CNNConfig


def build_mlp_model(input_dim: int, output_dim: int, config: MLPConfig = None) -> Sequential:
    """
    Build and compile a Multilayer Perceptron (MLP) model for multi-label classification.

    Parameters:
    - input_dim (int): Number of input features.
    - output_dim (int): Number of output labels.
    - config (MLPConfig): Configuration for MLP model.

    Returns:
    - model (Sequential): Compiled Keras model.
    """
    if config is None:
        config = MLPConfig(output_dim=output_dim)
    
    model = Sequential()
    model.add(Dense(config.hidden_layer_1, input_dim=input_dim, activation='relu'))
    model.add(Dropout(config.dropout_rate))
    model.add(Dense(config.hidden_layer_2, activation='relu'))
    model.add(Dropout(config.dropout_rate))
    model.add(Dense(output_dim, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_cnn_model(
    vocab_size: int, embedding_dim: int, max_len: int, output_dim: int,
    conv_filters: int = 128, conv_kernel_size: int = 5,
    dense_units: int = 128, dropout: float = 0.5
) -> Sequential:
    """
    Build and compile a CNN model for text classification.

    Parameters:
    - vocab_size (int): Size of vocabulary.
    - embedding_dim (int): Dimension of embeddings.
    - max_len (int): Maximum sequence length.
    - output_dim (int): Number of output labels.
    - conv_filters (int): Number of convolutional filters.
    - conv_kernel_size (int): Size of convolutional kernel.
    - dense_units (int): Number of units in dense layer.
    - dropout (float): Dropout rate.

    Returns:
    - model (Sequential): Compiled Keras model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
    model.add(Conv1D(conv_filters, conv_kernel_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
