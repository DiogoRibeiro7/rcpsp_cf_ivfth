"""
Entry point for ``python -m rcpsp_cf_ivfth.examples``.

Running the module file directly (``python -m rcpsp_cf_ivfth.examples.toy_instance``)
still works, but Python imports the package first and then re-executes the module,
which raises a RuntimeWarning about the double import. Going through ``__main__``
avoids that.
"""

from .toy_instance import run_toy_example

if __name__ == "__main__":
    run_toy_example()
