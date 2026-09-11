# GPU instance (Vast.ai) for braintrace

A rented Linux box where JAX binds CUDA by default. Code moves by git (commit, push, pull),
never by rsync. Nothing under `var/`, `.cache`, or `/h01` travels with git; re-fetch it.

## Rent

```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 disk_space>=128 reliability>0.98 inet_down>500 cuda_vers>=12.4 verified=true cpu_cores_effective>=16 cpu_ram>=32' --order 'dph_total' --limit 8
vastai create instance <OFFER_ID> --image ubuntu:22.04 --disk 128 --ssh --direct --label braintrace-gpu \
    --env '-e XLA_PYTHON_CLIENT_PREALLOCATE=false -e JAX_COMPILATION_CACHE_DIR=/workspace/jax-cache'
vastai show instance <ID> --raw   # ports["22/tcp"][0].HostPort is the direct SSH port
```

Use the plain `ubuntu:22.04` image. The `vastai/pytorch:*-auto` images did not accept the
account SSH key at all (tested 2026-09-11, including a reboot). CUDA arrives through pip
wheels (`jax[cuda12]`), so the image only needs the NVIDIA driver, which Vast mounts into
every container.

`~/.ssh/config` entry on the workstation (direct port, not the `sshN.vast.ai` proxy):

```
Host braintrace-gpu
  HostName <public_ipaddr>
  Port <direct 22/tcp port>
  User root
  IdentityFile ~/.ssh/<account key>
  ServerAliveInterval 60
  ServerAliveCountMax 60
```

## Provision (once per instance)

`scripts/provision_gpu_instance.sh` is idempotent and takes one step name or `all`:

| step | what it does |
|------|--------------|
| `deploy_key` | generates `~/.ssh/braintrace_deploy`, pins it for `github.com`, prints the public key. Register it: `gh repo deploy-key add key.pub --repo mithridatic/braintrace --title vast-braintrace-gpu --allow-write` |
| `clone` | `git clone git@github.com:mithridatic/braintrace.git /workspace/braintrace`, sets identity and `pull.rebase` |
| `venv` | uv-managed Python 3.14 at `/workspace/venv314` with the exact pins from `.github/containers/braintrace-gpu/Dockerfile`, then an editable `pip install -e .[cuda12,examples]`. Appends PATH and JAX env to `~/.bashrc`. Leaves `JAX_PLATFORMS` unset so CUDA is the default backend |
| `h01` | `fetch_h01('/h01')`, `fetch_h01_annotations('/h01')`, and the `proofread104.zip` symlink the probe expects |

```bash
scp scripts/provision_gpu_instance.sh braintrace-gpu:/root/provision.sh
ssh braintrace-gpu '/root/provision.sh deploy_key'   # then register the printed key
ssh braintrace-gpu '/root/provision.sh clone && /root/provision.sh venv && /root/provision.sh h01'
```

The editable install matters: the Docker images bake a copy of the repo at `/opt/braintrace`
that silently shadows the checkout. On the box the checkout is the only copy.

## Verify

```bash
ssh braintrace-gpu 'nvidia-smi --query-gpu=name,driver_version --format=csv'
ssh braintrace-gpu 'python -c "import jax; print(jax.devices())"'   # CudaDevice(id=0)
ssh braintrace-gpu 'cd /workspace/braintrace && git push --dry-run origin main'
ssh braintrace-gpu 'cd /workspace/braintrace && python -m examples.h01_arc_probe --cache /h01 --cells 12 --sparse-learning --output /evidence/h01-12-gpu-$(date +%Y%m%d).json'
```

The probe hashes source files by path relative to the repo root and imports
`examples.pp_prop.*`, so run it from `/workspace/braintrace`.

## Daily flow

Work on a branch on the box, commit, push. Pull on the workstation, or the reverse.
Stop the instance when idle (`vastai stop instance <ID>`); disk still bills while stopped,
and destroying loses the venv and `/h01` (about ten minutes to rebuild) but nothing pushed.
