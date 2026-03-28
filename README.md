## Description

Repository for data and scripts needed for reproducing analysis on Master's Thesis. This repo will be updated with more information.

## Dependencies

This project uses [mise](https://mise.jdx.dev/) to handle python language versions and [uv](https://docs.astral.sh/uv/) for python project management. Once you have installed mise, the `mise.toml` file will handle the setup of python and uv, cd into the directory and run:

```bash
mise install
```

Once uv is configured, the `pyproject.toml` file will configure virtual environments and python packages:

```bash
uv  sync
```
