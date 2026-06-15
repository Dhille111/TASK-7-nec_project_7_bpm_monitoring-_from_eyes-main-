from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def build_cnn():

    model = Sequential()

    model.add(
        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            input_shape=(300, 1)
        )
    )

    model.add(
        MaxPooling1D(
            pool_size=2
        )
    )

    model.add(
        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu"
        )
    )

    model.add(
        MaxPooling1D(
            pool_size=2
        )
    )

    model.add(
        Flatten()
    )

    model.add(
        Dense(
            128,
            activation="relu"
        )
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        Dense(
            64,
            activation="relu"
        )
    )

    model.add(
        Dense(
            1,
            activation="linear"
        )
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


if __name__ == "__main__":

    model = build_cnn()

    model.summary()