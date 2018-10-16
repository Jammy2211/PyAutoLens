FROM ubuntu:18.04
LABEL maintainer="Nan Li <linan7788626@gmail.com>"

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
ADD jupyter /home/user/.jupyter
ADD PyAutoLens /home/user/PyAutoLens
ADD entrypoint.sh /usr/local/bin/entrypoint.sh

# Setup ENV
EXPOSE 6006
EXPOSE 8888
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/local/lib/"
ENV PYTHONPATH "$PYTHONPATH:/home/user/PyAutoLens/"
WORKDIR "/home/user/PyAutoLens/workspace"

# Start up
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter-notebook", "--ip=0.0.0.0"]
#CMD ["/bin/bash"]
