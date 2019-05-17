FROM cv-server:0.3

# TODO: find a compatible version of pytorch that won't messup tensorflow initialize
RUN /root/miniconda/envs/eyeq/bin/pip install psutil
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/nmslib/hnswlib.git /nmslib
RUN /root/miniconda/envs/eyeq/bin/pip install pybind11
WORKDIR "/nmslib/python_bindings/"
RUN /root/miniconda/envs/eyeq/bin/python3 setup.py install
RUN conda install -n eyeq -y pytorch torchvision cuda92 -c pytorch
RUN /root/miniconda/envs/eyeq/bin/pip install mxnet-cu92
RUN conda install -n eyeq -y shapely=1.6.4=py35h7ef4460_0
###################################################################################################################

COPY source /ems-cv-services/source
COPY models /ems-cv-services/models
COPY data /ems-cv-services/data/

WORKDIR "/ems-cv-services/source"
ENV PYTHONPATH /ems-cv-services/source:$PYTHONPATH
ENV PRODUCTION True

ENTRYPOINT ["/root/miniconda/envs/eyeq/bin/python3", "ems_start_point.py"]
#ENTRYPOINT ["/root/miniconda/envs/eyeq/bin/python3", "test_server.py"]
