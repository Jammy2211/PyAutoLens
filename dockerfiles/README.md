***need more work***
Before Building the Images, please git pull to update PyAutoLens first.

- Build the Docker Image for PyAutoLens
	docker build -t linan7788626/docker-pyautolens .


- Run the image with PyAutoLens
	docker run -it -p 8888:8888 -p 6006:6006 -v /data/nanli/deepLearning/shared:/App/PyAutoLens/shared linan7788626/docker-pyautolens

	docker run -it -e LOCAL_USER_ID=`id -u $USER` -h PyAutoLens --name Run_PyAutoLens \
					-p 8888:8888 -p 6006:6006 \
					-v $HOME/pyautolens-shared:/App/PyAutoLens/shared \
					linan7788626/docker-pyautolens

	docker run -it -e LOCAL_USER_ID=`id -u $USER` -h PyAutoLens -p 8888:8888 -p 6006:6006 -v $HOME/pyautolens-shared:/home/user/PyAutoLens/shared linan7788626/docker-pyautolens

	docker run -it -e LOCAL_USER_ID=`id -u $USER` -h PyAutoLens -p 8888:8888 -p 6006:6006 -v $HOME/PyAutoLensWorkDir:/home/user/PyAutoLens/workDir linan7788626/docker-pyautolens


- Pull the Docker Image
	docker pull linan7788626/docker-pyautolens


- gosu keyservers


for server in ha.pool.sks-keyservers.net \
              hkp://p80.pool.sks-keyservers.net:80 \
              keyserver.ubuntu.com \
              hkp://keyserver.ubuntu.com:80 \
              pgp.mit.edu; do
    gpg --keyserver "$server" --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4 && break || echo "Trying new server..."
done

- gosu docker setup
	https://denibertovic.com/posts/handling-permissions-with-docker-volumes/
