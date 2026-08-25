ARG BASE_IMAGE=braintrace-gpu:0.11.0-py314-msgspec
FROM ${BASE_IMAGE}

ARG BRAINTRACE_SOURCE_COMMIT=uncommitted

LABEL org.opencontainers.image.title="BrainTrace Example 21 BrainCell ARC runtime" \
      org.opencontainers.image.version="0.11.0-py314-arc" \
      org.opencontainers.image.revision="${BRAINTRACE_SOURCE_COMMIT}"

COPY . /opt/braintrace

ENV PYTHONPATH=/opt/braintrace

RUN python -m pip install --no-cache-dir --no-deps "/opt/braintrace[cuda12,examples]"

RUN python -m pip install --no-cache-dir "braincell==0.1.0"

COPY --from=arc_data . /datasets/arc/raw

WORKDIR /opt/braintrace

CMD ["python", "/opt/braintrace/examples/pp_prop/21-braincell-arc.py", "--help"]
