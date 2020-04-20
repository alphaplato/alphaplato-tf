FROM tensorflow/serving

MAINTAINER plato.sg.lee@gmail.com

COPY java /usr/local/java
COPY hadoop /root/hadoop

ENV JAVA_HOME /usr/local/java
ENV HADOOP_HOME /root/hadoop
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${JAVA_HOME}/jre/lib/amd64/server

RUN echo '#!/bin/bash \n\n\
source ${HADOOP_HOME}/libexec/hadoop-config.sh \n \
CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob) tensorflow_model_server --port=8500 --rest_api_port=9000 \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

EXPOSE 8500
EXPOSE 9000

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]