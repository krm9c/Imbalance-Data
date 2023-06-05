import tensorflow as tf


class model_nn:
    def __init__(self, config, input, feature):

        self.config = config
        self.inputs = input
        self.features = feature

        self.model = self.model_construct()

    def model_construct(self):
        x = tf.keras.layers.Dense(self.config["units"], activation="tanh")(self.features)
        # x = tf.keras.layers.Dense(config["units"], activation=config["activation"])(all_features)
        x = tf.keras.layers.Dropout(0.3)(x)
        # x = tf.keras.layers.Dropout(config["dropout_rate"])(x)

        for i in range(self.config["num_layers"]):
            x = tf.keras.layers.Dense(self.config["units"], activation="tanh")(x)
            # x = tf.keras.layers.Dense(config["units"], activation=config["activation"])(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            # x = tf.keras.layers.Dropout(config["dropout_rate"])(x)
            # print("new layer")

        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.Model(self.inputs, output)

    def model_compile(self, optimizer=None, loss=None, metric=None):

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        if loss is None:
            loss = "binary_crossentropy"

        if metric is None:
            metric = ["accuracy"]

        self.model.compile(optimizer, loss, metrics=metric)

    def model_train(self, train_data, val_data):

        return self.model.fit(train_data, validation_data=val_data, epochs=self.config["num_epochs"], verbose=0)
