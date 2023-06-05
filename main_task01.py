import pandas as pd
import deephyper
import ray

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup
from tensorflow.python.client import device_lib

from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import EqualsCondition
from deephyper.evaluator import Evaluator
from deephyper.search.hps import CBO


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


n_gpus = len(get_available_gpus())
if n_gpus > 1:
    n_gpus -= 1

is_gpu_available = n_gpus > 0

if is_gpu_available:
    print(f"{n_gpus} GPU{'s are' if n_gpus > 1 else ' is'} available.")
else:
    print("No GPU available")


def load_data():
    file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
    dataframe = pd.read_csv(file_url)

    val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
    train_dataframe = dataframe.drop(val_dataframe.index)

    return train_dataframe, val_dataframe


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = tf.keras.layers.Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = (
        tf.keras.layers.StringLookup if is_string else tf.keras.layers.IntegerLookup
    )
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


def run_function(config: dict):
    tf.autograph.set_verbosity(0)
    # Load data and split into validation set
    train_dataframe, val_dataframe = load_data()
    train_ds = dataframe_to_dataset(train_dataframe)
    val_ds = dataframe_to_dataset(val_dataframe)
    # train_ds = train_ds.batch(config["batch_size"])
    # val_ds = val_ds.batch(config["batch_size"])
    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    # Categorical features encoded as integers
    sex = tf.keras.Input(shape=(1,), name="sex", dtype="int64")
    cp = tf.keras.Input(shape=(1,), name="cp", dtype="int64")
    fbs = tf.keras.Input(shape=(1,), name="fbs", dtype="int64")
    restecg = tf.keras.Input(shape=(1,), name="restecg", dtype="int64")
    exang = tf.keras.Input(shape=(1,), name="exang", dtype="int64")
    ca = tf.keras.Input(shape=(1,), name="ca", dtype="int64")

    # Categorical feature encoded as string
    thal = tf.keras.Input(shape=(1,), name="thal", dtype="string")

    # Numerical features
    age = tf.keras.Input(shape=(1,), name="age")
    trestbps = tf.keras.Input(shape=(1,), name="trestbps")
    chol = tf.keras.Input(shape=(1,), name="chol")
    thalach = tf.keras.Input(shape=(1,), name="thalach")
    oldpeak = tf.keras.Input(shape=(1,), name="oldpeak")
    slope = tf.keras.Input(shape=(1,), name="slope")

    all_inputs = [
        sex,
        cp,
        fbs,
        restecg,
        exang,
        ca,
        thal,
        age,
        trestbps,
        chol,
        thalach,
        oldpeak,
        slope,
    ]

    # Integer categorical features
    sex_encoded = encode_categorical_feature(sex, "sex", train_ds, False)
    cp_encoded = encode_categorical_feature(cp, "cp", train_ds, False)
    fbs_encoded = encode_categorical_feature(fbs, "fbs", train_ds, False)
    restecg_encoded = encode_categorical_feature(restecg, "restecg", train_ds, False)
    exang_encoded = encode_categorical_feature(exang, "exang", train_ds, False)
    ca_encoded = encode_categorical_feature(ca, "ca", train_ds, False)

    # String categorical features
    thal_encoded = encode_categorical_feature(thal, "thal", train_ds, True)

    # Numerical features
    age_encoded = encode_numerical_feature(age, "age", train_ds)
    trestbps_encoded = encode_numerical_feature(trestbps, "trestbps", train_ds)
    chol_encoded = encode_numerical_feature(chol, "chol", train_ds)
    thalach_encoded = encode_numerical_feature(thalach, "thalach", train_ds)
    oldpeak_encoded = encode_numerical_feature(oldpeak, "oldpeak", train_ds)
    slope_encoded = encode_numerical_feature(slope, "slope", train_ds)

    all_features = tf.keras.layers.concatenate(
        [
            sex_encoded,
            cp_encoded,
            fbs_encoded,
            restecg_encoded,
            exang_encoded,
            slope_encoded,
            ca_encoded,
            thal_encoded,
            age_encoded,
            trestbps_encoded,
            chol_encoded,
            thalach_encoded,
            oldpeak_encoded,
        ]
    )
    x = tf.keras.layers.Dense(config["units"], activation="tanh")(all_features)
    # x = tf.keras.layers.Dense(config["units"], activation=config["activation"])(all_features)
    x = tf.keras.layers.Dropout(0.3)(x)
    # x = tf.keras.layers.Dropout(config["dropout_rate"])(x)

    for i in range(config["num_layers"]):
        x = tf.keras.layers.Dense(config["units"], activation="tanh")(x)
        # x = tf.keras.layers.Dense(config["units"], activation=config["activation"])(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        # x = tf.keras.layers.Dropout(config["dropout_rate"])(x)
        # print("new layer")

    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(all_inputs, output)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer, "binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        train_ds, epochs=config["num_epochs"], validation_data=val_ds, verbose=0
    )

    return history.history["val_accuracy"][-1]


# ...
# history = model.fit(
#     train_ds, epochs=config["num_epochs"], validation_data=val_ds, verbose=0
# )
# return history.history["val_accuracy"][-1]
# ...

from deephyper.problem import HpProblem

# Creation of an hyperparameter problem
problem = HpProblem()

# Discrete hyperparameter (sampled with uniform prior)
problem.add_hyperparameter((8, 128), "units", default_value=32)
problem.add_hyperparameter((10, 100), "num_epochs", default_value=50)

# Categorical hyperparameter (sampled with uniform prior)
# ACTIVATIONS = [
#     "elu", "gelu", "hard_sigmoid", "linear", "relu", "selu",
#     "sigmoid", "softplus", "softsign", "swish", "tanh",
# ]
# problem.add_hyperparameter(ACTIVATIONS, "activation", default_value="relu")

# Real hyperparameter (sampled with uniform prior)
# problem.add_hyperparameter((0.0, 0.6), "dropout_rate", default_value=0.5)

# Discrete and Real hyperparameters (sampled with log-uniform)
# problem.add_hyperparameter((8, 256, "log-uniform"), "batch_size", default_value=32)
# problem.add_hyperparameter((1e-5, 1e-2, "log-uniform"), "learning_rate", default_value=1e-3)
problem.add_hyperparameter((1, 4, "log-uniform"), "num_layers", default_value=2)

problem

# We launch the Ray run-time depending of the detected local ressources
# and execute the `run` function with the default configuration
# WARNING: in the case of GPUs it is important to follow this scheme
# to avoid multiple processes (Ray workers vs current process) to lock
# the same GPU.
if is_gpu_available:
    if not (ray.is_initialized()):
        ray.init(num_cpus=n_gpus, num_gpus=n_gpus, log_to_driver=False)

    run_default = ray.remote(num_cpus=1, num_gpus=1)(run)
    objective_default = ray.get(run_default.remote(problem.default_configuration))
else:
    if not (ray.is_initialized()):
        ray.init(num_cpus=1, log_to_driver=False)
    run_default = run_function
    objective_default = run_default(problem.default_configuration)

print(f"Accuracy Default Configuration:  {objective_default:.3f}")
method = "thread"
method_kwargs = {
    "num_workers": 4,
    "callbacks": [TqdmCallback()]
}
if is_gpu_available:
    method_kwargs["num_cpus"] = n_gpus
    method_kwargs["num_gpus"] = n_gpus
    method_kwargs["num_cpus_per_task"] = 1
    method_kwargs["num_gpus_per_task"] = 1

evaluator = Evaluator.create(run_function, method, method_kwargs)

# configs = [{"units": 8, ...}, ...]
# evaluator.submit(configs)
# ...
# # To collect the first finished task (asynchronous)
# tasks_done = evaluator.get("BATCH", size=1)

# # To collect all of the pending tasks (synchronous)
# tasks_done = evaluator.get("ALL")

from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback


def get_evaluator(run_function):
    # Default arguments for Ray: 1 worker and 1 worker per evaluation
    method_kwargs = {
        "num_cpus": 1,
        "num_cpus_per_task": 1,
        "callbacks": [TqdmCallback()]
    }

    # If GPU devices are detected then it will create 'n_gpus' workers
    # and use 1 worker for each evaluation
    if is_gpu_available:
        method_kwargs["num_cpus"] = n_gpus
        method_kwargs["num_gpus"] = n_gpus
        method_kwargs["num_cpus_per_task"] = 1
        method_kwargs["num_gpus_per_task"] = 1

    evaluator = Evaluator.create(
        run_function,
        method="ray",
        method_kwargs=method_kwargs
    )
    # evaluator = Evaluator.create(
    #     run_function,
    #     # method="serial",
    #     method="thread"
    #     # method_kwargs=method_kwargs
    # )
    print(
        f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}", )

    return evaluator


print("Evaluator 1")
evaluator_1 = get_evaluator(run_function)

# Uncomment the following line to show the arguments of CBO.
# help(CBO)

# Instanciate the search with the problem and the evaluator that we created before

search = CBO(problem, evaluator_1, initial_points=[problem.default_configuration])

results = search.search(max_evals=100)
results

i_max = results.objective.argmax()
best_config = results.iloc[i_max][:-3].to_dict()

print(f"The default configuration has an accuracy of {objective_default:.3f}. \n"
      f"The best configuration found by DeepHyper has an accuracy {results['objective'].iloc[i_max]:.3f}, \n"
      f"discovered after {results['m:timestamp_gather'].iloc[i_max]:.2f} secondes of search.\n")

best_config

# # Create a new evaluator
# print("Evaluator 2")
# evaluator_2 = get_evaluator(run_function)
#
# # Create a new AMBS search with strong explotation (i.e., small kappa)
# search_from_checkpoint = CBO(problem, evaluator_2, kappa=0.001)
#
# # Initialize surrogate model of Bayesian optization (in AMBS)
# # With results of previous search
# search_from_checkpoint.fit_surrogate(results)
# results_from_checkpoint = search_from_checkpoint.search(max_evals=10)
#
# i_max = results_from_checkpoint.objective.argmax()
# best_config = results_from_checkpoint.iloc[i_max][:-3].to_dict()
#
# print(f"The default configuration has an accuracy of {objective_default:.3f}. "
#       f"The best configuration found by DeepHyper has an accuracy {results_from_checkpoint['objective'].iloc[i_max]:.3f}, "
#       f"finished after {results_from_checkpoint['m:timestamp_gather'].iloc[i_max]:.2f} secondes of search.")
#
# best_config
#
# if best_config.get("dense_2", False):
#     x = tf.keras.layers.Dense(config["dense_2:units"], activation=config["dense_2:activation"])(x)
#
#
# def run_with_condition(config: dict):
#     tf.autograph.set_verbosity(0)
#
#     train_dataframe, val_dataframe = load_data()
#
#     train_ds = dataframe_to_dataset(train_dataframe)
#     val_ds = dataframe_to_dataset(val_dataframe)
#
#     train_ds = train_ds.batch(config["batch_size"])
#     val_ds = val_ds.batch(config["batch_size"])
#
#     # Categorical features encoded as integers
#     sex = tf.keras.Input(shape=(1,), name="sex", dtype="int64")
#     cp = tf.keras.Input(shape=(1,), name="cp", dtype="int64")
#     fbs = tf.keras.Input(shape=(1,), name="fbs", dtype="int64")
#     restecg = tf.keras.Input(shape=(1,), name="restecg", dtype="int64")
#     exang = tf.keras.Input(shape=(1,), name="exang", dtype="int64")
#     ca = tf.keras.Input(shape=(1,), name="ca", dtype="int64")
#
#     # Categorical feature encoded as string
#     thal = tf.keras.Input(shape=(1,), name="thal", dtype="string")
#
#     # Numerical features
#     age = tf.keras.Input(shape=(1,), name="age")
#     trestbps = tf.keras.Input(shape=(1,), name="trestbps")
#     chol = tf.keras.Input(shape=(1,), name="chol")
#     thalach = tf.keras.Input(shape=(1,), name="thalach")
#     oldpeak = tf.keras.Input(shape=(1,), name="oldpeak")
#     slope = tf.keras.Input(shape=(1,), name="slope")
#
#     all_inputs = [
#         sex,
#         cp,
#         fbs,
#         restecg,
#         exang,
#         ca,
#         thal,
#         age,
#         trestbps,
#         chol,
#         thalach,
#         oldpeak,
#         slope,
#     ]
#
#     # Integer categorical features
#     sex_encoded = encode_categorical_feature(sex, "sex", train_ds, False)
#     cp_encoded = encode_categorical_feature(cp, "cp", train_ds, False)
#     fbs_encoded = encode_categorical_feature(fbs, "fbs", train_ds, False)
#     restecg_encoded = encode_categorical_feature(restecg, "restecg", train_ds, False)
#     exang_encoded = encode_categorical_feature(exang, "exang", train_ds, False)
#     ca_encoded = encode_categorical_feature(ca, "ca", train_ds, False)
#
#     # String categorical features
#     thal_encoded = encode_categorical_feature(thal, "thal", train_ds, True)
#
#     # Numerical features
#     age_encoded = encode_numerical_feature(age, "age", train_ds)
#     trestbps_encoded = encode_numerical_feature(trestbps, "trestbps", train_ds)
#     chol_encoded = encode_numerical_feature(chol, "chol", train_ds)
#     thalach_encoded = encode_numerical_feature(thalach, "thalach", train_ds)
#     oldpeak_encoded = encode_numerical_feature(oldpeak, "oldpeak", train_ds)
#     slope_encoded = encode_numerical_feature(slope, "slope", train_ds)
#
#     all_features = tf.keras.layers.concatenate(
#         [
#             sex_encoded,
#             cp_encoded,
#             fbs_encoded,
#             restecg_encoded,
#             exang_encoded,
#             slope_encoded,
#             ca_encoded,
#             thal_encoded,
#             age_encoded,
#             trestbps_encoded,
#             chol_encoded,
#             thalach_encoded,
#             oldpeak_encoded,
#         ]
#     )
#     x = tf.keras.layers.Dense(config["units"], activation=config["activation"])(
#         all_features
#     )
#
#     ### START - NEW LINES
#     if config.get("dense_2", False):
#         x = tf.keras.layers.Dense(config["dense_2:units"], activation=config["dense_2:activation"])(x)
#     ### END - NEW LINES
#
#     x = tf.keras.layers.Dropout(config["dropout_rate"])(x)
#     output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
#     model = tf.keras.Model(all_inputs, output)
#
#     optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
#     model.compile(optimizer, "binary_crossentropy", metrics=["accuracy"])
#
#     history = model.fit(
#         train_ds, epochs=config["num_epochs"], validation_data=val_ds, verbose=0
#     )
#
#     return history.history["val_accuracy"][-1]
#
#
# # Define the hyperparameter problem
# problem_with_condition = HpProblem()
#
# # Define the same hyperparameters as before
# problem_with_condition.add_hyperparameter((8, 128), "units")
# problem_with_condition.add_hyperparameter(ACTIVATIONS, "activation")
# problem_with_condition.add_hyperparameter((0.0, 0.6), "dropout_rate")
# problem_with_condition.add_hyperparameter((10, 100), "num_epochs")
# problem_with_condition.add_hyperparameter((8, 256, "log-uniform"), "batch_size")
# problem_with_condition.add_hyperparameter((1e-5, 1e-2, "log-uniform"), "learning_rate")
#
# # Add a new hyperparameter "dense_2 (bool)" to decide if a second fully-connected layer should be created
# hp_dense_2 = problem_with_condition.add_hyperparameter([True, False], "dense_2")
# hp_dense_2_units = problem_with_condition.add_hyperparameter((8, 128), "dense_2:units")
# hp_dense_2_activation = problem_with_condition.add_hyperparameter(ACTIVATIONS, "dense_2:activation")
#
# problem_with_condition.add_condition(EqualsCondition(hp_dense_2_units, hp_dense_2, True))
# problem_with_condition.add_condition(EqualsCondition(hp_dense_2_activation, hp_dense_2, True))
# problem_with_condition
#
# print("Evaluator 3")
# evaluator_3 = get_evaluator(run_with_condition)
#
# search_with_condition = CBO(problem_with_condition, evaluator_3)
# results_with_condition = search_with_condition.search(max_evals=10)
#
# results_with_condition
#
# i_max = results_with_condition.objective.argmax()
# best_config = results_with_condition.iloc[i_max][:-3].to_dict()
#
# print(f"The default configuration has an accuracy of {objective_default:.3f}. "
#       f"The best configuration found by DeepHyper has an accuracy {results_with_condition['objective'].iloc[i_max]:.3f}, "
#       f"finished after {results_with_condition['m:timestamp_gather'].iloc[i_max]:.2f} seconds of search.")
#
# best_config
