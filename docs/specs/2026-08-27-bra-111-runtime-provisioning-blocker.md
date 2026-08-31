# BRA-111 runtime provisioning record

## Result

The assigned execution environment is not qualified to run the dependent
acceptance workload. No product source or test assertion was modified.

## Environment observed

| Component | Observed state |
| --- | --- |
| Python | `Python 3.12.3` at `/usr/bin/python3.12` |
| pip | unavailable (`No module named pip`) |
| ensurepip | unavailable (`No module named ensurepip`) |
| virtual environments | creation fails because `ensurepip` is unavailable |
| pytest | unavailable on `PATH` |
| JAX | unavailable (cannot install) |
| brainstate | unavailable (cannot install) |
| system packages | `python3-pip`, `python3.12-venv`, and `python3-venv` are not installed |

The package candidates advertised by APT were `python3-pip 24.0+dfsg-1ubuntu1.3`
and `python3.12-venv 3.12.3-1ubuntu0.16`, but the container blocks privileged
installation (`sudo` is disabled by `no new privileges`).  Bootstrapping pip
from PyPI is also unavailable because DNS resolution for `bootstrap.pypa.io`
and `pypi.org` fails.

## Control-plane bridge

The queue transport metadata exists, but the run-scoped loopback listener is
absent. Restarting the supplied bridge program on its assigned port failed with
`listen EPERM: operation not permitted`. Consequently, the normal issue update
request fails before connecting; no bridge token or credential was recorded.

## Reverification

The continuation run repeated the prerequisite checks without changing product
code or test assertions:

- `python3.12 --version` reported `Python 3.12.3`.
- `python3.12 -m pip --version` and `python3.12 -m ensurepip --version` both
  failed because their modules are absent.
- Import discovery reported `pytest`, `jax`, and `brainstate` as unavailable.
- No executable Python environment or cached wheels for those packages was
  found in the available runtime paths.
- The injected Paperclip API URL and key variables were present, but its
  loopback endpoint refused connections. No listener process was running.

The dependent issue's exact focused collection remains unavailable through the
control plane while this bridge is down; this record therefore cannot identify
or run a substitute command.

## Required remediation and handoff

The runtime/harness operator must provide either:

1. a prebuilt Python 3.12 environment with `pytest`, `jax`, and `brainstate`,
   or package/bootstrap access sufficient to install them; and
2. a permitted, live run-scoped Paperclip bridge listener on the injected
   loopback endpoint.

After both are present, wake the dependent acceptance owner and rerun its exact
Git, package-version, and focused pytest collection commands. This record does
not substitute for that acceptance run.
