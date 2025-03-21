FROM python:3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf8

WORKDIR /app

# Copy files and finish install
ADD ./install-models-and-data.py .
ADD ./sklearn_crfsuite-0.3.6-py2.py3-none-any.whl .
ADD ./requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt \
    && pip install sklearn_crfsuite-0.3.6-py2.py3-none-any.whl \
    && python install-models-and-data.py 

EXPOSE 8888

WORKDIR /jupyter-data

ENTRYPOINT ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", \
    "--allow-root", "--NotebookApp.token=Geslo.01", "--notebook-dir=/jupyter-data"]