ARG BASE_IMAGE=braintrace-gpu:0.11.0-py314-msgspec
FROM ${BASE_IMAGE}

ARG BRAINTRACE_SOURCE_COMMIT=uncommitted
ARG ARC_AGI_1_COMMIT=unknown
ARG ARC_TRAINING_EXCLUSION=42939e035eb673b934fe566f18d92a374bf3edda5b29c491667f0cc5773e451d

LABEL org.opencontainers.image.title="BrainTrace Example 21 ARC runtime" \
      org.opencontainers.image.version="0.11.0-py314-arc" \
      org.opencontainers.image.revision="${BRAINTRACE_SOURCE_COMMIT}" \
      org.braintrace.dataset.arc-agi-1.revision="${ARC_AGI_1_COMMIT}" \
      org.braintrace.dataset.arc-agi-1.training-tasks="399" \
      org.braintrace.dataset.arc-agi-1.evaluation-tasks="400"

COPY . /opt/braintrace

ENV PYTHONPATH=/opt/braintrace

RUN python -m pip install --no-cache-dir --no-deps "/opt/braintrace[cuda12,examples]"

COPY --from=arc_data . /datasets/arc/raw

RUN python /opt/braintrace/examples/pp_prop/build_arc_dataset_index.py \
      --dataset-root /datasets/arc/raw \
      --output-root /datasets/arc/index \
      --manifest-path /datasets/arc/example21-sources.json \
      --version "${ARC_AGI_1_COMMIT}" \
      --training-exclusion "${ARC_TRAINING_EXCLUSION}" \
    && python -c "import msgspec; from examples.pp_prop.latent_workspace_task import DatasetSource, assert_no_evaluation_leakage, load_dataset_source; payload = msgspec.json.decode(open('/datasets/arc/example21-sources.json', 'rb').read()); loaded = tuple(load_dataset_source(DatasetSource(**item)) for item in payload['sources']); assert tuple(len(item.tasks) for item in loaded) == (399, 400); assert_no_evaluation_leakage(item.manifest for item in loaded)"

ENV EXAMPLE21_SOURCE_MANIFEST=/datasets/arc/example21-sources.json

WORKDIR /opt/braintrace

CMD ["python", "/opt/braintrace/examples/pp_prop/21-arc-agi-latent-reasoning.py", "--help"]
