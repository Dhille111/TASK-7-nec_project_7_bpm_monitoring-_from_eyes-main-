from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def build_bilstm():

    model = Sequential()

    model.add(

        Bidirectional(

            LSTM(
                64,
                return_sequences=True
            ),

            input_shape=(300,1)

        )

    )

    model.add(

        Dropout(
            0.3
        )

    )

    model.add(

        Bidirectional(

            LSTM(
                32
            )

        )

    )

    model.add(

        Dense(
            32,
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

    model = build_bilstm()

    model.summary()