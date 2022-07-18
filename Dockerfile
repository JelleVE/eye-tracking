FROM ubuntu:20.04

# Set timezone:
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

RUN apt update -y && apt-get install -y \
		build-essential \
		cmake \
		ffmpeg \
		libsm6 \
		libxext6 \
		locales \
		python3-pip \
	&& rm -rf /var/lib/apt/lists/*

# Fix ascii to utf8 => https://webkul.com/blog/setup-locale-python3/
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


# Setup app
RUN mkdir /app
RUN rm -rf ~/.cache/pip
WORKDIR /app
ADD . /app/

EXPOSE 5000

# Setup dependencies
WORKDIR /app/
RUN apt-get install -y python3-pip
RUN python3 --version
RUN pip3 install --upgrade pip
RUN pip3 install -r build/requirements_general.txt --no-cache-dir
RUN pip3 install -r build/requirements_torch.txt --no-cache-dir

WORKDIR /app/PIPNet/FaceBoxesV2/utils
RUN chmod +x make.sh
RUN sh make.sh

WORKDIR /app
RUN chmod +x /app/run.sh
CMD ["/app/run.sh"]