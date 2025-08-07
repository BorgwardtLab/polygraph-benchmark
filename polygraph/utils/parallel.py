# -*- coding: utf-8 -*-
"""parallel.py

This script contains utility functions for parallel processing.
"""

import contextlib
import time
from typing import Any, Callable, Generator, Iterable, List

import joblib
from joblib import Parallel, delayed
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.console import Console

console = Console()


def flatten_lists(lists: List[List[Any]]) -> List[Any]:
    flattened = []
    for sublist in lists:
        if isinstance(sublist, list):
            flattened.extend(sublist)
        else:
            flattened.append(sublist)
    return flattened


def make_batches(lst: List[Any], n: int) -> List[List[Any]]:
    chunks = [lst[i : i + n] for i in range(0, len(lst), n)]
    return chunks


@contextlib.contextmanager
def rich_joblib(
    progress: Progress, task_id: TaskID
) -> Generator[None, None, None]:
    """Context manager to patch joblib to report into Rich progress bar."""

    class RichBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.completed_batches = 0

        def __call__(self, *args: Any, **kwargs: Any) -> None:
            self.completed_batches += 1
            progress.update(
                task_id,
                advance=self.batch_size,
                refresh=True,
            )
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = RichBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback


def create_progress_bar(show_progress: bool = True) -> Progress:
    """Creates a standardized progress bar with preset columns."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not show_progress,
    )


def distribute_function(
    func: Callable,
    X: Iterable,
    n_jobs: int,
    description: str = "",
    total: int = 1,
    use_enumerate: bool = False,
    show_progress: bool = True,
    **kwargs,
) -> Any:
    if total == 1:
        total = len(X)  # type: ignore

    if n_jobs == 1:
        with create_progress_bar(show_progress) as progress:
            task_id = progress.add_task(description, total=total)
            results = []
            if use_enumerate:
                for idx, x in enumerate(X):
                    result = func(idx, x, **kwargs)
                    results.append(result)
                    progress.update(task_id, advance=1, refresh=True)
            else:
                for x in X:
                    result = func(x, **kwargs)
                    results.append(result)
                    progress.update(task_id, advance=1, refresh=True)
        return results

    if use_enumerate:
        parallel_execution = (
            delayed(func)(idx, x, **kwargs) for idx, x in enumerate(X)
        )
    else:
        parallel_execution = (delayed(func)(x, **kwargs) for x in X)

    if show_progress:
        with create_progress_bar(show_progress) as progress:
            task_id = progress.add_task(description, total=total)
            with rich_joblib(progress, task_id):
                Xt = Parallel(n_jobs=n_jobs)(parallel_execution)
    else:
        Xt = Parallel(n_jobs=n_jobs)(parallel_execution)

    return Xt


def retry(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator that retries a function if it raises an exception.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Time to wait between retries in seconds
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


def batched_distribute_function(
    func: Callable,
    X: List[Any],
    n_jobs: int,
    description: str = "",
    total: int = 1,
    use_enumerate: bool = False,
    show_progress: bool = True,
    batch_size: int = 100,
    **kwargs,
) -> Any:
    """
    Note: func must be able to iterate over the batches.
    """
    batches = make_batches(X, batch_size)
    return flatten_lists(
        distribute_function(
            func,
            batches,
            n_jobs,
            description,
            total,
            use_enumerate,
            show_progress,
            **kwargs,
        )
    )
