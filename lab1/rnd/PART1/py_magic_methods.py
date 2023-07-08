from typing import Any

class MyClass:
    def __getitem__(self, __name: str) -> Any:
        return f'You`ve tried to read property `{__name}` of me'
    def __call__(self):
        print(f'You`ve tried to invoke me')

obj = MyClass()
print('Accessing object state via []')
print(obj['somestuff'])
print()
print('Invoking class instance')
obj()
