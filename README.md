# build-mamba

Pre-built Linux wheels for [Mamba](https://github.com/state-spaces/mamba), across Python,
PyTorch, CUDA, and CPU architectures.

## Installation

Following the PyTorch convention, artifacts are published to a separate index for each CUDA
version. Each wheel has a local version suffix that identifies the CUDA and PyTorch versions it was
built against, such as `mamba-ssm==2.3.2.post1+cu.12.8.torch.2.11`, and requires the matching PyTorch
release.

Pre-built wheels are available on [Astral's GPU indexes](https://wheels.astralshosted.com/index.html).
For example, to install a CUDA 12.8 build:

```console
$ uv add mamba-ssm --index astral-cu128=https://wheels.astralshosted.com/simple/cu128/
```

This configures the index and uses it as the source for `mamba-ssm`:

```toml
[tool.uv.sources]
mamba-ssm = { index = "astral-cu128" }

[[tool.uv.index]]
name = "astral-cu128"
url = "https://wheels.astralshosted.com/simple/cu128/"
```

Or, with `uv pip`:

```console
$ uv pip install --index https://wheels.astralshosted.com/simple/cu128/ mamba-ssm
```

## Supported versions

Wheels are available for the following `mamba-ssm` versions:

- [`2.3.2.post1`](https://github.com/astral-sh-build/build-mamba/releases/tag/v2.3.2.post1)
- [`2.3.1`](https://github.com/astral-sh-build/build-mamba/releases/tag/v2.3.1)
- [`2.3.0`](https://github.com/astral-sh-build/build-mamba/releases/tag/v2.3.0-r1)
- [`2.2.6.post3`](https://github.com/astral-sh-build/build-mamba/releases/tag/v2.2.6.post3-r2)

The latest release, Mamba 2.3.2.post1, supports the following combinations:

| PyTorch | Python    | `x86_64` CUDA          | `aarch64` CUDA         |
| ------- | --------- | ---------------------- | ---------------------- |
| 2.4.1   | 3.9–3.12  | 12.1, 12.4             | —                      |
| 2.5.1   | 3.9–3.12  | 12.1, 12.4             | —                      |
| 2.6.0   | 3.9–3.12  | 12.4, 12.6             | 12.6                   |
| 2.7.1   | 3.9–3.13  | 12.6, 12.8             | 12.8                   |
| 2.8.0   | 3.9–3.13  | 12.6, 12.8, 12.9       | 12.9                   |
| 2.9.1   | 3.10–3.14 | 12.6, 12.8, 12.9, 13.0 | 12.6, 12.8, 12.9, 13.0 |
| 2.10.0  | 3.10–3.14 | 12.6, 12.8, 12.9, 13.0 | 12.6, 12.8, 12.9, 13.0 |
| 2.11.0  | 3.10–3.14 | 12.6, 12.8, 12.9, 13.0 | 12.6, 12.8, 12.9, 13.0 |

## License

build-mamba is licensed under the [Apache License, Version 2.0](LICENSE).

<div align="center">
  <a target="_blank" href="https://astral.sh" style="background:none">
    <img src="https://raw.githubusercontent.com/astral-sh/ruff/main/assets/svg/Astral.svg" alt="Made by Astral">
  </a>
</div>
