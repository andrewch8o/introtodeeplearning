from typing import Any


class MyClass:
    def __getitem__(self, __name: str) -> Any:
        return f'You`ve tried to index me by property {__name}'
    def __call__(self, **kwargs):
        print(f'You`ve tried to invoke me with properties {kwargs}')

o = MyClass()

print(o['test'])
o(a=1, b=2)