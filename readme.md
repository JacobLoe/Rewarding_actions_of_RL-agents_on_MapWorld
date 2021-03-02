mkdir data_tf1.5
docker volume create --driver local --opt type=none --opt device=/home/jacob/Downloads/IM/data_tf1.5 --opt o=bind data_tf1.5
docker build -f Dockerfile_tf1.5 -t jacobloe/tf1.5:0.1 .
docker run --rm -it -v data_tf1.5:/root/data --gpus all --entrypoint /bin/bash --name tf1.5 jacobloe/tf1.5:0.1

docker build -f Dockerfile_tf1.15.2 -t jacobloe/tf1.15.2:0.1 .
docker run --rm -it -v data_tf1.5:/root/data --gpus all --entrypoint /bin/bash --name tf1.5 jacobloe/tf1.15.2:0.1
