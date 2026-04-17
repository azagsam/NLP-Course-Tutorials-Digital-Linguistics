FROM python:3.11.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf8

WORKDIR /app

# Copy files and finish install
ADD ./requirements.txt .

RUN apt-get update && apt-get install -y graphviz
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 8888

WORKDIR /jupyter-data

ENTRYPOINT ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", \
    "--allow-root", "--NotebookApp.token=Geslo.01", "--notebook-dir=/jupyter-data"]