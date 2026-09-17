#!/usr/bin/env bash
# Provision a fresh Vast.ai box for braintrace: deploy key, uv + Python 3.14 venv with the
# Dockerfile pins, editable install, H01 cache. Idempotent; safe to re-run.
set -euo pipefail

STEP="${1:-all}"

deploy_key() {
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  if [ ! -f ~/.ssh/braintrace_deploy ]; then
    ssh-keygen -t ed25519 -N "" -f ~/.ssh/braintrace_deploy -C "vast-braintrace-gpu" >/dev/null
  fi
  if ! grep -q "braintrace_deploy" ~/.ssh/config 2>/dev/null; then
    printf 'Host github.com\n  IdentityFile ~/.ssh/braintrace_deploy\n  IdentitiesOnly yes\n' >> ~/.ssh/config
  fi
  grep -q github.com ~/.ssh/known_hosts 2>/dev/null || ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
  echo "PUBKEY: $(cat ~/.ssh/braintrace_deploy.pub)"
}

clone() {
  mkdir -p /workspace && cd /workspace
  if [ ! -d /workspace/braintrace/.git ]; then
    git clone git@github.com:mithridatic/braintrace.git
  fi
  cd /workspace/braintrace
  git config user.name "Special Officer Doofy"
  git config user.email "tracey.blume@netincolor.com"
  git config pull.rebase true
  git remote -v; git log --oneline -1
}

venv() {
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv python install 3.14
  [ -x /workspace/venv314/bin/python ] || uv venv --python 3.14 /workspace/venv314
  PY=/workspace/venv314/bin/python
  # Pins copied verbatim from .github/containers/braintrace-gpu/Dockerfile
  uv pip install --python "$PY" \
      "numpy==2.4.6" "scikit-learn==1.9.0" "psutil==7.2.2" "hypothesis==6.165.2" \
      "pytest==9.1.1" "pytest-cov==7.1.0" "pytest-xdist==3.8.0" "pytest-timeout==2.4.0" \
      "msgspec==0.21.1" "brainstate==0.5.3" "brainunit==0.5.2" "brainevent[cuda12]==0.2.1" \
      "braintools==0.3.0" "braincell==0.1.0" "optax==0.2.8" "brainpy==2.8.2" \
      "brainpy-state==0.1.0" "jax[cuda12]==0.11.0" "numba==0.67.0" \
      "matplotlib" "mypy>=1.8" "build"
  uv pip install --python "$PY" --no-deps -e "/workspace/braintrace[cuda12,examples]"
  "$PY" -c "import sys, jax; assert sys.version_info[:2]==(3,14); assert jax.__version__=='0.11.0'; print(jax.devices())"
  if ! grep -q venv314 ~/.bashrc; then
    cat >> ~/.bashrc <<'EOF'
# braintrace GPU environment
export PATH=/workspace/venv314/bin:$HOME/.local/bin:$PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/workspace/jax-cache
EOF
  fi
  mkdir -p /workspace/jax-cache
}

h01() {
  cd /workspace/braintrace
  /workspace/venv314/bin/python -c "
from braintrace.datasets.h01 import fetch_h01
from braintrace.datasets.h01_annotations import fetch_h01_annotations
fetch_h01('/h01'); fetch_h01_annotations('/h01')"
  [ -e /h01/proofread104.zip ] || ln -s /h01/104_proofread_neurons_swc.zip /h01/proofread104.zip
  ls -la /h01
}

case "$STEP" in
  deploy_key) deploy_key ;;
  clone) clone ;;
  venv) venv ;;
  h01) h01 ;;
  all) deploy_key; clone; venv; h01 ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
