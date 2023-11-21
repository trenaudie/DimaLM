# %%
import random
from dataclasses import dataclass, field
from typing import List
import time
from asyncio import sleep


@dataclass
class Message:
    content: str


@dataclass
class Choice:
    message: Message = field(default_factory=Message)


@dataclass
class Response:
    choices: List[Choice] = field(default_factory=list)


class Completions:
    def __init__(self, time_sleep: int) -> None:
        self.time_sleep = time_sleep

    def create(
        self,
        model: str,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ""},
        ],
        max_tokens=200,
        top_p=1.0,
        temperature=0.5,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["14.", " 14."],
    ):
        return_txt = ""
        for k in range(4, 20):
            up_down_not_relevant = random.choice(["UP", "DOWN", "NOT RELEVANT"])
            line = f"{k}. {up_down_not_relevant}\n"
            return_txt += line
            if "13." in line:
                break
        print(f"sleeping for {self.time_sleep} seconds")
        time.sleep(self.time_sleep)
        return Response(choices=[Choice(message=Message(content=return_txt))])


class Chat:
    def __init__(self, time_sleep=2.5):
        self.time_sleep = time_sleep
        self.completions = Completions(self.time_sleep)


class OpenAIFake:
    def __init__(self, time_sleep=2.5):
        self.time_sleep = time_sleep
        self.chat = Chat(self.time_sleep)


# %%
if __name__ == "__main__":
    client = OpenAIFake()
    response = client.chat.completions.create()
    content = response.choices[0].message.content
    print(content)
