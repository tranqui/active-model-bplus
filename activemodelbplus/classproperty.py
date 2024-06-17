#!/usr/bin/env python3

class classproperty:
    """Decorator for class properties from https://stackoverflow.com/a/76301341."""

    def __init__(self, func):
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)