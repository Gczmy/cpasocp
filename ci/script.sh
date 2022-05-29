#!/bin/bash
set -euxo pipefail

regular_test() {
    # Run Python tests
    # ------------------------------------

    # --- create virtual environment
    export PYTHONPATH=.

    # --- install virtualenv
    pip install virtualenv

    # --- create virtualenv
    virtualenv -p python3.8 venv

    # --- activate venv
    source venv/bin/activate

    # --- upgrade pip within venv
    pip install --upgrade pip

    # --- install raocp-toolbox
    pip install .

    # --- run the tests
    # export PYTHONPATH=.
    python -W ignore tests/test_sets.py -v
    python -W ignore tests/test_linear_operator.py -v
    python -W ignore tests/test_online_part.py -v 
}


main() {
    regular_test
}

main
