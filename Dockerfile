FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV PYTHONFAULTHANDLER=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

LABEL name="brainformr"

RUN apt-get update && \
	apt-get install -qqy \
	git \
	&& apt-get clean && \ 
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python -m pip install --upgrade pip && \
	python -m pip install git@https://github.com/alexj-lee/brainformr.git  && \
	find . -type d -name __pycache__ -exec rm -r {} + && \
	find . -type f -name "*.pyc" -exec rm -r {} + && \
	rm -rf /root/.cache