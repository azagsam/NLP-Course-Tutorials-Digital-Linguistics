## Build your own image and run it (Docker)

1. Build image

```
docker buildx build \
  --platform linux/amd64 \
  -t nlp-course-lab8 \
  ./docker_environment
```

2. Run container

```
docker run --platform linux/amd64 -it \
  --mount type=bind,source=$(pwd),target=/jupyter-data \
  -p 8888:8888 \
  nlp-course-lab8
```

3. Navigate to http://localhost:8888 (password is 'Geslo.01') and enjoy!