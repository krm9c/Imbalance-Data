from deephyper.evaluator.callback import TqdmCallback
from tensorflow.python.client import device_lib
from deephyper.problem import HpProblem
import ray


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


def get_problem():
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

    return problem


def set_run_method(gpu_available, n_gpu=1, run_function=None, problem=None, method=None, method_kwargs=None):
    if gpu_available:
        if not (ray.is_initialized()):
            ray.init(num_cpus=n_gpu, num_gpus=n_gpu, log_to_driver=False)

        run_default = ray.remote(num_cpus=1, num_gpus=1)(run_function())
        objective_default = ray.get(run_default.remote(problem.default_configuration))
    else:
        if not (ray.is_initialized()):
            ray.init(num_cpus=1, log_to_driver=False)
        run_default = run_function
        objective_default = run_default(problem.default_configuration)

    print(f"Accuracy Default Configuration:  {objective_default:.3f}")

    if method is None:
        method = "thread"

    if method_kwargs is None:
        method_kwargs = {
            "num_workers": 4,
            "callbacks": [TqdmCallback()]
        }
        if gpu_available:
            method_kwargs["num_cpus"] = n_gpu
            method_kwargs["num_gpus"] = n_gpu
            method_kwargs["num_cpus_per_task"] = 1
            method_kwargs["num_gpus_per_task"] = 1

    return method, method_kwargs, objective_default


def get_evaluator(run_function, evaluator, gpu_available=False, n_gpu=1):

    # Default arguments for Ray: 1 worker and 1 worker per evaluation
    method_kwargs = {
        "num_cpus": 1,
        "num_cpus_per_task": 1,
        "callbacks": [TqdmCallback()]
    }

    # If GPU devices are detected then it will create 'n_gpus' workers
    # and use 1 worker for each evaluation
    if gpu_available:
        method_kwargs["num_cpus"] = n_gpu
        method_kwargs["num_gpus"] = n_gpu
        method_kwargs["num_cpus_per_task"] = 1
        method_kwargs["num_gpus_per_task"] = 1

    evaluator = evaluator.create(
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
