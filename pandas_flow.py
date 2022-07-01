from copy import deepcopy
from functools import wraps
from time import time
from typing import List

import pandas as pd


class Task:
    def __init__(self, task_name: str = "", steps: List[str] = [], inputs_name: List[str] = [],
                 outputs_name: List[str] = [], output_name: str = "", verbose: bool = False):
        self.steps = steps
        self.inputs_name = inputs_name
        self.outputs_name = outputs_name
        self.task_name = task_name
        self.output_name = output_name
        self.verbose = verbose

    def reset(self):
        self.steps = []
        self.inputs_name = []
        self.outputs_name = []
        self.task_name = ""
        self.output_name = ""
        self.verbose = False


class PandasFlow:
    #TODO Observer pattern: object with messages
    global task
    task = Task()

    def __init__(self, pipeline, *, task_name, output_name, verbose=False):
        self.pipeline = pipeline
        self.task_name = task_name
        task.reset()
        task.output_name = output_name
        task.verbose = verbose

    def __enter__(self):
        return self

    @staticmethod
    def wrap_task(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time()
            result = func(*args, **kwargs)
            if task.verbose:
                print("-" * 10 + func.__name__.upper() + "-" * 10)
                print(f"Dataframe shape before: {args[0].shape}")
                print(f"Dataframe shape after: {result.shape}")
                process_time = round(time() - start_time, 3)
                print(f"Process time: {process_time} seconds")
            task.steps.append(func.__name__)

            args_df = [a for a in list(args) + list(kwargs.values()) if isinstance(a, pd.DataFrame)]
            for df in args_df:
                if hasattr(df, "name") and df.name not in [None, task.output_name]:
                    task.inputs_name.append(df.name)

            for key, value in kwargs.items():
                if key == "table" and value != task.output_name:
                    task.inputs_name.append(value)

            task.inputs_name = sorted(list(set(task.inputs_name)))

            result.name = task.output_name
            return result

        return wrapper

    def __exit__(self, type, value, traceback):
        task.outputs_name.append(task.output_name)
        task.task_name = self.task_name
        self.pipeline.add_task(deepcopy(task))


class Pipeline:
    def __init__(self):
        self.pipeline = []
        self.graph = {}

    def add_task(self, task):
        self.pipeline.append(task)

    def generate_graph(self):
        for task in self.pipeline:
            graph_tmp = {}
            graph_tmp["inputs_name"] = task.inputs_name
            graph_tmp["outputs_name"] = task.outputs_name
            graph_tmp["steps"] = task.steps
            graph_tmp["next_tasks"] = [k.task_name for k in self.pipeline if
                                       any(t in k.inputs_name for t in task.outputs_name)]
            self.graph[task.task_name] = graph_tmp
        return self.graph