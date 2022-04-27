#!/usr/bin/env python3

import sys

# Use @cache and @cached_property decorators for caching directly in memory without disk.
from functools import cache, cached_property

# cloudpickle package has convenient pickling tool that works with compiled sympy expressions.
# We can use this to cache compiled expressions to the disk.
import cloudpickle

class DiskCache:
    def __init__(self, function, path, verbose):
        self.function = function
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.load()
        self.verbose = verbose

    def load(self):
        try:
            with open(self.path, 'rb') as f:
                self.cache = cloudpickle.load(f)
        except FileNotFoundError:
            self.cache = {}

    def save(self):
        with open(self.path, 'wb') as f:
            f.write(cloudpickle.dumps(self.cache))

    def __call__(self, *args, **kwargs):
        signature = (args, tuple(sorted(kwargs.items())))
        key = cloudpickle.dumps(signature)
        if key in self.cache:
            return self.cache[key]

        result = self.function(*args, **kwargs)
        self.cache[key] = result
        self.save()

        if self.verbose:
            sys.stderr.write('caching call to %s with arguments: %r\n' % (self.function.__qualname__, signature))

        return result

class DiskCacheFolder:
    def __init__(self, cachedir, verbose=False):
        self.cachedir = cachedir
        self.verbose = verbose

    def __call__(self, function):
        module = function.__module__
        if module == '__main__':
            module = Path(__file__).resolve().stem

        path = self.cachedir / module / function.__qualname__
        return DiskCache(function, path, self.verbose)

# Create directory in the temp folder for caching to the disk.
from pathlib import Path
import tempfile
source_path = Path(__file__).resolve()
source_dir = source_path.parent
source_project = source_dir.parts[-1]
cachedir = Path('%s/%s' % (tempfile.gettempdir(), source_project))

# Use @disk_cache decorator for caching to disk.
disk_cache = DiskCacheFolder(cachedir, verbose=True)
