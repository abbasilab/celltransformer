FROM nvidia/cuda:12.1.0-base-ubuntu20.04 AS build

WORKDIR /celltransformer

ARG PYTHON_VERSION=3.10.14
ARG UV_VERSION=0.6.3

COPY . /celltransformer

RUN apt-get update && \
	apt-get install -qqy \
	git \
	build-essential \
	curl 

ADD https://astral.sh/uv/${UV_VERSION}/install.sh /install.sh
RUN chmod +x /install.sh && \
	/install.sh && \
	mv /root/.local/bin/uv /usr/local/bin/uv && \
	uv python install ${PYTHON_VERSION} --python-preference managed

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    UV_PYTHON=python${PYTHON_VERSION} 

RUN	/usr/local/bin/uv venv --python ${PYTHON_VERSION} --python-preference managed /opt/venv && \
	. /opt/venv/bin/activate && \
	/usr/local/bin/uv pip install setuptools wheel && \
	/usr/local/bin/uv pip install /celltransformer --no-build-isolation 

FROM nvidia/cuda:12.1.0-base-ubuntu20.04 AS runtime

ARG PYTHON_VERSION
ENV PYTHONFAULTHANDLER=1 \
	PYTHONUNBUFFERED=1

LABEL name="celltransformer"

COPY --from=build /opt/venv /opt/venv
RUN --mount=from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \ 
	uv python install 3.10.14 --python-preference managed

ENV PATH="/opt/venv/bin:$PATH" 