# BrainTrace temporal-credit GPU image

Build from the BrainTrace repository root so the image records the exact source
under test:

```powershell
$commit = git rev-parse HEAD
docker build --file .github/containers/braintrace-gpu/Dockerfile `
  --build-arg BRAINTRACE_SOURCE_COMMIT=$commit `
  --tag braintrace-gpu:0.11.0-py314 .
```

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
