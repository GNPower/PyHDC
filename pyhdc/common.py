"""The common module contains common functions and classes used by the other modules."""


def hello_world():
    """Prints "Hello World!" to the console."""
    print("Hello World!")


def hello(name):
    """Prints "Hello {name}!" to the console.

    Args:
        name (str): The name to print.
    """
    print(f"Hello {name}!")


def add(a, b):
    """Adds two numbers.

    Args:
        a (int): Number a.
        b (int): Number b.
    Returns:
        (int): sum
    """
    return a + b
