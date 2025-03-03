# Building docs

The docs are available [here](https://on-point-rnd.github.io/EBES/).

To build docs manually, run from the repo root
```bash
pip install -r requirements-dev.txt
make html
```
If `make` is not installed, run manually:
```bash
sphinx-build -M html docs build
```
The static HTML docs will appear in the `build/html` folder.
