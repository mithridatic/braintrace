FROM python:3.14.0-slim-trixie@sha256:5af4c7f950774a1abf3fd4e8e3fc95f4d0fe684c7fdc1eb10777fc5a017371c7

LABEL org.opencontainers.image.title="BrainTrace temporal-credit GPU runtime" \
      org.opencontainers.image.version="0.11.0-py314"

ENV JAX_PLATFORMS="" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir \
      "numpy==2.4.6" \
      "scikit-learn==1.9.0" \
      "psutil==7.2.2" \
      "hypothesis==6.165.2" \
      "pytest==9.1.1" \
      "pytest-cov==7.1.0" \
      "pytest-xdist==3.8.0" \
      "pytest-timeout==2.4.0" \
      "msgspec==0.21.1" \
      "brainstate==0.5.3" \
      "brainunit==0.5.2" \
      "brainevent[cuda12]==0.2.1" \
      "braintools==0.3.0" \
      "optax==0.2.8" \
      "brainpy==2.8.2" \
      "brainpy-state==0.1.0" \
      "jax[cuda12]==0.11.0" \
      "numba==0.67.0"

RUN apt-get update \
    && apt-get install --no-install-recommends --yes git \
    && rm -rf /var/lib/apt/lists/* \
    && git --version

COPY . /opt/braintrace

RUN python -m pip install --no-cache-dir --no-deps "/opt/braintrace[cuda12,examples]" \
    && python -c "import jax, sys; assert sys.version_info[:2] == (3, 14); assert jax.__version__ == '0.11.0'"

ARG BRAINTRACE_SOURCE_COMMIT=uncommitted

LABEL org.opencontainers.image.revision="${BRAINTRACE_SOURCE_COMMIT}"

# Persistent XLA compilation cache. Declared after the install layers so editing
# it never invalidates the pip download cache. Bind-mount a shared host directory
# at /cache/jax so kernels compiled by one run are reused by later runs and by
# concurrent containers; left unmounted it is container-local, i.e. the previous
# cold-compile-every-run behaviour.
ENV JAX_COMPILATION_CACHE_DIR=/cache/jax \
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.5 \
    JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0

RUN mkdir -p /cache/jax

WORKDIR /work

CMD ["python", "examples/pp_prop/17-temporal-credit-benchmark.py", "--help"]
