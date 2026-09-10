Use pytest for unit tests.

Place unit tests and unit test data in `tests/`.

Mirror the source directory structure for where to place tests.

If a module (e.g., `foo.py`) requires many tests, instead create a subdirectory (e.g., `foo/`) to contain multiple testing modules that are split up into logical chuncks.

Some tests will require expensive, shared setup. Use setup and teardown tools to efficiently reuse immutable inputs to tests.

For tests add terse docstrings explaining what they test. If there are relevant specifics include those too. Be terse.

For testing helper functions and testing helper classes, add terse docstrings explaing what they do and why they are useful.
