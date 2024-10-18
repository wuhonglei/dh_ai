import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import tensorflow as tf
from tensorflow import keras

from src.utils import timer

layers = keras.layers
models = keras.models


@timer("dl")
def dl(data: pd.DataFrame) -> pd.DataFrame:

    data["Keyword"] = data["Keyword"].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        data["Keyword"],
        data["Category"],
        random_state=0,
        test_size=0.2,
    )
    max_words = 1000
    tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)
    tokenize.fit_on_texts(x_train)  # fit tokenizer to our training text data
    x_train = tokenize.texts_to_matrix(x_train)
    x_test = tokenize.texts_to_matrix(x_test)

    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    num_classes = np.max(y_train) + 1
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    batch_size = 32
    epochs = 2
    drop_ratio = 0.5

    model = models.Sequential()
    model.add(layers.Dense(512, input_shape=(max_words,)))
    model.add(layers.Activation("relu"))
    # model.add(layers.Dropout(drop_ratio))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
    )

    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
