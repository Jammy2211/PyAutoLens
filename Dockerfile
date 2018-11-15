FROM ubuntu:18.04
LABEL authors="Nan Li <linan7788626@gmail.com>, Richard Hayes <richard@rghsoftware.co.uk>"

# Install packages with apt-get
RUN apt-get update \
	&& apt-get install -yq --no-install-recommends \
			gpg gpg-agent dirmngr ca-certificates curl \
			openmpi-bin openmpi-common \
			gcc g++ gfortran cmake make git \
			liblapack3 liblapack-dev \
			python3-pip python3-setuptools python3-pytest \
	&& apt-get clean \
	&& apt-get autoclean \
	&& apt-get autoremove \
	&& rm -rf /var/lib/apt/lists/* \
	&& ln -sf /usr/bin/python3 /usr/bin/python \
	&& git clone https://github.com/JohannesBuchner/MultiNest.git \
	&& cd /MultiNest/build/ \
	&& cmake .. \
	&& make \
	&& make install \
	&& rm -fr /MultiNest

# Install packages with pip
RUN pip3 --no-cache-dir install \
			pymultinest \
			numba \
			scipy \
			astropy \
			scikit-learn \
			jupyter \
			matplotlib \
			colorama \
			docopt \
			getdist==0.2.8.4.2

# Set up permissions.
RUN gpg --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4 \
	&& curl -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.10/gosu-$(dpkg --print-architecture)" \
    && curl -o /usr/local/bin/gosu.asc -SL "https://github.com/tianon/gosu/releases/download/1.10/gosu-$(dpkg --print-architecture).asc" \
    && gpg --verify /usr/local/bin/gosu.asc \
    && rm /usr/local/bin/gosu.asc \
    && chmod +x /usr/local/bin/gosu

# Copy files
ADD dockerfiles/jupyter /home/user/.jupyter
ADD autolens /home/user/autolens
ADD workspace /home/user/workspace_temp
ADD dockerfiles/entrypoint.sh /usr/local/bin/entrypoint.sh
ADD dockerfiles/bashrc /home/user/.bashrc
ENV SHELL "/bin/bash"

# Setup ENV
EXPOSE 6006
EXPOSE 8888
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/local/lib/"
ENV PYTHONPATH "$PYTHONPATH:/home/user/"
WORKDIR "/home/user/workspace"

# Start up
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter-notebook", "--ip=0.0.0.0"]
