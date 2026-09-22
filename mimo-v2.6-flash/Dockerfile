FROM vllm/vllm-openai:v0.29.0
ARG B12X_REV=59d51a36a942d56a9c36265855cdc7856fa7712e
ADD https://github.com/voipmonitor/b12x/archive/${B12X_REV}.tar.gz /tmp/b12x.tar.gz
RUN mkdir /opt/b12x-src && tar -xzf /tmp/b12x.tar.gz --strip-components=1 -C /opt/b12x-src \
    && python3 -m pip install --no-deps /opt/b12x-src && rm /tmp/b12x.tar.gz
COPY patches/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY SOURCE_LOCK.json /opt/mimo-source-lock.json
RUN python3 -c "import json,hashlib,pathlib; m=json.load(open('/opt/mimo-source-lock.json')); root=pathlib.Path('/usr/local/lib/python3.12/dist-packages'); [(_ for _ in ()).throw(AssertionError(p)) if hashlib.sha256((root/v['target']).read_bytes()).hexdigest()!=v['sha256'] else None for p,v in m['files'].items()]"
