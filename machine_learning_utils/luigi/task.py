from functools import partial
from luigi import Task, LocalTarget


class Requirement:

    def __init__(self, task_class, **params):
        self.task_class = task_class
        self.params = params

    def __get__(self, task, cls):
        if task is None:
            return self
        return task.clone(
            self.task_class,
            **self.params)


class Requires:
    """Composition to replace :meth:`luigi.task.Task.requires`
    Example::
        class MyTask(Task):
            # Replace task.requires()
            requires = Requires()
            other = Requirement(OtherTask)
            def run(self):
                # Convenient access here...
                with self.other.output().open('r') as f:
                    ...
        MyTask().requires()
        {'other': OtherTask()}
    """

    def __get__(self, task, cls):
        if task is None:
            return self
        # Bind self/task in a closure
        return partial(self.__call__, task)

    def __call__(self, task):
        """Returns the requirements of a task
        Assumes the task class has :class:`.Requirement` descriptors, which
        can clone the appropriate dependencies from the task instance.
        :returns: requirements compatible with `task.requires()`
        :rtype: dict
        """
        task_dict = {k: getattr(task, k) for k in dir(task)}
        return {k: v for k, v in task_dict.items() if isinstance(getattr(type(task), k, None), Requirement)}


class TargetOutput:
    """Composition to replace :meth:`luigi.task.Task.output"""

    def __init__(self, file_pattern='{task.__class__.__name__}', ext='.txt', target_class=LocalTarget, **target_kwargs):
        self.file_pattern = file_pattern
        self.ext = ext
        self.target_class = target_class
        self.target_kwargs = target_kwargs

    def __get__(self, task, cls):
        if task is None:
            return self
        return partial(self.__call__, task)

    def __call__(self, task):
        path = 'data/' + self.file_pattern.format(task=task) + self.ext
        #glob = self.file_pattern.format(task=task) + self.ext
        return self.target_class(path,  **self.target_kwargs)
