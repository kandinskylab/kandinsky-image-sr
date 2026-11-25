# PyTorch build for this repo's own environment. `auto` lets uv detect the installed NVIDIA
# driver and take torch from the matching PyTorch wheel index (CPU build without a GPU);
# every other package comes from PyPI. Force a build for your driver if needed:
#   just torch_backend=cu130 setup
torch_backend := "auto"

# Environments are installed with `uv pip`, not synced from uv.lock, so `uv run` must not
# re-sync them back to the lockfile's PyPI torch.
export UV_NO_SYNC := "1"

# Create .venv and install the package with dev tools.
# Extras are positional, e.g. FlashAttention-2 (Linux + CUDA, needs nvcc): just setup flash
setup *EXTRAS:
    @uv venv --allow-existing --quiet
    @{{just_executable()}} install .venv/bin/python {{EXTRAS}}

# Install into the ACTIVE conda environment instead of .venv (no venv is created).
# Extras are positional: just setup-conda flash
setup-conda *EXTRAS:
    @if [ -z "${CONDA_PREFIX:-}" ]; then echo "Error: no conda environment is active. Activate one first (e.g. 'conda activate myenv')." && exit 1; fi
    @echo "=== Installing kandinsky_image_sr into conda env $CONDA_PREFIX ==="
    @{{just_executable()}} install "$CONDA_PREFIX/bin/python" {{EXTRAS}}

# torch is always re-resolved, so an existing build for the wrong CUDA gets replaced.
# Extras go in a second step: flash-attn builds without isolation against the torch
# that the first step installed.
[private]
install python *EXTRAS:
    #!/usr/bin/env bash
    set -euo pipefail
    uv pip install --python "{{python}}" --torch-backend "{{torch_backend}}" --reinstall-package torch -e . --group dev
    if [[ -n "{{EXTRAS}}" ]]; then
        uv pip install --python "{{python}}" --torch-backend "{{torch_backend}}" -e ".[{{replace(EXTRAS, ' ', ',')}}]" --group dev
    fi

lint:
    @uv run ruff check .
    @uv run ruff format --check .

fmt:
    @uv run ruff format .
    @uv run ruff check --fix .

test *ARGS:
    @uv run pytest {{ARGS}}

# Remove generated Python, test, lint, and package-build artifacts.
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    find . -path ./.venv -prune -o -type d \( -name '__pycache__' -o -name '.ruff_cache' -o -name '.pytest_cache' -o -name 'build' -o -name 'dist' -o -name '*.egg-info' \) -prune -exec rm -rf {} +
    find . -path ./.venv -prune -o -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '.coverage' -o -name '.coverage.*' \) -delete
