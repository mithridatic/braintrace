# BrainTrace temporal-credit GPU image

Build from the BrainTrace repository root so the image records the exact source
under test:

```powershell
$commit = git rev-parse HEAD
docker build --file .github/containers/braintrace-gpu/Dockerfile `
  --build-arg BRAINTRACE_SOURCE_COMMIT=$commit `
  --tag braintrace-gpu:0.11.0-py314 .
```

## Always mount the XLA compilation cache

The image sets `JAX_COMPILATION_CACHE_DIR=/cache/jax`, but that path is
container-local: a `--rm` run with nothing mounted over it discards the cache on
exit and the next run recompiles and re-autotunes every kernel from scratch.
XLA's GPU backend picks kernels by benchmarking candidate variants on the device,
so this is the dominant cost of a cold run.

Prefer Compose. `docker-compose.yml` in the repository root declares the cache
as a named volume, so the mount happens with no flags to remember:

```bash
docker compose run --rm gpu python examples/pp_prop/18-structural-evolution.py
```

Concurrent `docker compose run` invocations share the one volume and warm each
other. Measured on a 512x512 tanh-matmul jit: 3.160 s cold, 0.019 s warm.

For one-off runs outside Compose there are wrappers that supply the same mount:

```powershell
.github\containers\braintrace-gpu\run-gpu-container.ps1 `
  -Mount "${PWD}:/work" -- python examples/pp_prop/18-structural-evolution.py
```

```bash
bash .github/containers/braintrace-gpu/run-gpu-container.sh \
  --mount "$PWD:/work" -- python examples/pp_prop/18-structural-evolution.py
```

The PowerShell wrapper requires `--` before the in-container command, otherwise
PowerShell binds the first token to `-CacheDirectory`. The wrappers default to a
shared host cache (`%LOCALAPPDATA%\braintrace\jax-cache` /
`~/.cache/braintrace/jax-cache`) and accept `-Env`/`--env` and
`-WorkDir`/`--workdir` for the provenance variables the benchmark drivers
require.

Raw `docker run` remains valid, but nothing mounts the cache for you: add
`--volume "<hostCache>:/cache/jax"` yourself or the run compiles cold.

The base is pinned to the linux/amd64 manifest for Python 3.14.0 slim-trixie.
JAX/JAXlib are pinned to 0.11.0 and the reviewed scientific dependencies are
exactly versioned. A benchmark run must pass the host-observed image ID through
`BRAINTRACE_IMAGE_DIGEST`; the result fingerprint records it together with the
driver, CUDA version, device, source commit, dirty state, and package versions.

Example development smoke:

```powershell
$image = "braintrace-gpu:0.11.0-py314"
$digest = docker image inspect $image --format '{{.Id}}'
$commit = git rev-parse HEAD
docker run --rm --gpus all `
  --env BRAINTRACE_IMAGE_DIGEST=$digest `
  --env BRAINTRACE_SOURCE_COMMIT=$commit `
  --env BRAINTRACE_SOURCE_DIRTY=true `
  --env NVIDIA_DRIVER_VERSION=595.79 `
  --env CUDA_VERSION=12.9 `
  --volume "${PWD}:/work" `
  $image python examples/pp_prop/17-temporal-credit-benchmark.py `
    --device gpu --horizon short --updates 2 --neurons 24 --degree 4 --allow-dirty
```

The source mount is intentional for development. Sealed evidence must build
from and run a clean accepted commit, and the result source commit must match
the image revision label.

## Development hyperparameter searches

Build a dedicated image after the search code is verified. Running from the
baked `/opt/braintrace` tree makes the local image ID an immutable fingerprint
of the dirty development snapshot, so resumed raw results cannot silently mix
source revisions:

```powershell
$image = "braintrace-gpu:0.11.0-py314-devsearch"
$commit = git rev-parse HEAD
docker build --file .github/containers/braintrace-gpu/Dockerfile `
  --build-arg BRAINTRACE_SOURCE_COMMIT=$commit --tag $image .
$digest = docker image inspect $image --format '{{.Id}}'
$gainResults = Join-Path $PWD "temp/temporal-credit-gain-search"

docker run --rm --gpus all `
  --env BRAINTRACE_IMAGE_DIGEST=$digest `
  --env BRAINTRACE_SOURCE_COMMIT=$commit `
  --env BRAINTRACE_SOURCE_DIRTY=true `
  --env NVIDIA_DRIVER_VERSION=595.79 `
  --env CUDA_VERSION=12.9 `
  --volume "${gainResults}:/results" `
  --workdir /opt/braintrace `
  $image python examples/pp_prop/temporal_benchmark_gain_search.py `
    --output-directory /results `
    --container-image-digest $digest `
    --source-commit $commit `
    --device gpu
```

After the gain search completes, use its winner for the learning-rate search:

```powershell
$gain = (Get-Content "$gainResults/winner.json" -Raw | ConvertFrom-Json).winner.gain
$optimizerResults = Join-Path $PWD "temp/temporal-credit-optimizer-search"
docker run --rm --gpus all `
  --env BRAINTRACE_IMAGE_DIGEST=$digest `
  --env BRAINTRACE_SOURCE_COMMIT=$commit `
  --env BRAINTRACE_SOURCE_DIRTY=true `
  --env NVIDIA_DRIVER_VERSION=595.79 `
  --env CUDA_VERSION=12.9 `
  --volume "${optimizerResults}:/results" `
  --workdir /opt/braintrace `
  $image python examples/pp_prop/temporal_benchmark_search.py `
    --output-directory /results --device gpu --gain $gain
```

Both drivers use only the three fixed development bundles and validation
metrics. They reject sealed metrics, validate the image and source provenance
of every resumed file, and stage raw JSON atomically before promotion.
