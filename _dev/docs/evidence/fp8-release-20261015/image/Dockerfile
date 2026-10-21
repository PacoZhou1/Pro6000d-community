FROM vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b

LABEL org.opencontainers.image.title="Pro6000d Qwen3.6 FP8 release 20261015" \
      org.opencontainers.image.base.name="vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b" \
      io.pro6000d.scope="SM120 FP8 V2 + GDN F1/F2 mdNone fix + draft64K; target BF16 head unchanged"

COPY payload/ /opt/pro6000d/

ENV PRO6000D_RELEASE_V2=1 \
    PRO6000D_RELEASE_F=1 \
    PRO6000D_RELEASE_DRAFT64K=1 \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

RUN python3 -I /opt/pro6000d/install_bootstrap.py
RUN python3 -I /opt/pro6000d/verify_install.py --matrix

# Inherit the base ENTRYPOINT ["vllm", "serve"], Cmd=null and WorkingDir.
# No host instrument/subset mount or modified vLLM file is required.
