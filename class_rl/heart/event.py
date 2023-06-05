import tensorflow as tf
from class_rl.heart.model import model_nn
from class_rl.heart.dataloader import data_encoder


def run_function(config: dict):
    print("print config file")
    print(config)
    tf.autograph.set_verbosity(0)

    inputs, features, train, valid = data_encoder()

    model = model_nn(config, input=inputs, feature=features)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.model_compile(optimizer=optimizer, loss="binary_crossentropy", metric=["accuracy"])

    history = model.model_train(train_data=train, val_data=valid)

    return history.history["val_accuracy"][-1]