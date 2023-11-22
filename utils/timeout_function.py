# %%

import multiprocessing
from time import time
from pathlib import Path

ROOT_DIR = next(
    filter(lambda s: "LLM" in s.name, Path.cwd().iterdir().__next__().parents), None
)
import sys

sys.path.append(str(ROOT_DIR))
from data_preprocess.data_v6.fake_openai import OpenAIFake


def wrap_function(function, queue: multiprocessing.Queue, *args, **kwargs):
    try:
        result = function(*args, **kwargs)
        queue.put((True, result))
    except Exception as e:
        queue.put((False, e))


class Timeout:
    def __init__(self, function, limit: float):
        self._limit = limit
        self._process = multiprocessing.Process()
        self._queue = multiprocessing.Queue(1)
        self._function = function

    def wrap_function(self, function, queue: multiprocessing.Queue, *args, **kwargs):
        try:
            result = function(*args, **kwargs)
            queue.put((True, result))
        except Exception as e:
            queue.put((False, e))

    def __call__(self, *args, **kwargs):
        # starts the process
        self._process = multiprocessing.Process(
            target=self.wrap_function,
            args=(self._function, self._queue, *args),
            kwargs=kwargs,
            daemon=True,
        )
        self._process.start()
        self._timeout = self._limit + time()

    def is_ready(self):
        if self._queue.qsize() > 0:
            return True
        elif time() > self._timeout:
            if self._process.is_alive():
                self._process.terminate()
            raise TimeoutError("timeout")
        else:
            return False

    def get_value(self):
        # if not ready, then pass
        # if ready, then return value
        # if timeout then raise exception
        if hasattr(self, "value"):
            return self.value
        if not self.is_ready():
            return None
        else:
            self.success, self.value = self._queue.get()
            if not self.success:
                raise self.value
            return self.get_value()


def TimeoutWrapper(limit: float):
    def wrapper(function):
        return Timeout(function, limit)

    return wrapper


if __name__ == "__main__":
    # here si the api
    # example 1
    def example1():
        @TimeoutWrapper(3)
        def function2(raise_error: int = 0):
            from time import sleep

            sleep(2)
            if raise_error == 1:
                raise ValueError("test")
            else:
                return "test"

        args = [
            0,
        ]
        function2(*args)
        while True:
            try:
                value = function2.get_value()
                if value:
                    print(f"success {value}")
                    break
            except TimeoutError:
                print(f"timed out - fail")
                break

    client = OpenAIFake()
    client = OpenAIFake(3)
    function = client.chat.completions.create
    args = (
        "gpt-3.5-turbo-1106",
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "1.\nNews:\ncompany acquires shares in Telecom industries worth $3B\nResponse:UP."
                + "2.\nNews:\ncompany acquires shares in aerospace manufacturuing worth $3B\nResponse:",
            },
        ],
        150,
    )
    kwargs = {
        "stop": ["3.", " 3."],
    }
    function_with_timeout = TimeoutWrapper(5)(function)
    function_with_timeout(*args, **kwargs)
    while True:
        try:
            value = function_with_timeout.get_value()
            if value:
                print(f"success {value}")
                break
        except TimeoutError:
            print(f"timed out - fail")
            break


# %%
