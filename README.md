# cv-labs
[![CI](https://github.com/ShkalikovOleh/cv-labs/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ShkalikovOleh/cv-labs/actions/workflows/ci.yml)

Computer vision university labs

## Usage

You can use the docker image to interact with cv-labs and especially with examples.

First of all, pull image:

```docker pull ghcr.io/shkalikovoleh/cv-labs:master```

Run image and expose 8888 port:

```docker run -p 8888:8888 ghcr.io/shkalikovoleh/cv-labs:master```

Follow link stars with `127.0.0.1:8888/?token=` to access Jupyter Notebooks(examples) in your browser.
