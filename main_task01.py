from deephyper.evaluator import Evaluator
from deephyper.search.hps import CBO

from class_rl.heart.utils import get_available_gpus, get_problem, set_run_method, get_evaluator
from class_rl.heart.event import run_function


n_gpus = len(get_available_gpus())
if n_gpus > 1:
    n_gpus -= 1

is_gpu_available = n_gpus > 0

if is_gpu_available:
    print(f"{n_gpus} GPU{'s are' if n_gpus > 1 else ' is'} available.")
else:
    print("No GPU available")

problem = get_problem()
method, method_kwargs, objective_default = set_run_method(is_gpu_available, n_gpus, run_function, problem)

evaluator = Evaluator.create(run_function, method, method_kwargs)

print("Start running evaluator 1:")
evaluator_1 = get_evaluator(run_function, evaluator, is_gpu_available, n_gpus)

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
