FROM continuumio/miniconda3
# 0. main directory
WORKDIR /app
# 1. copy environment
COPY environment.yaml .
# 2. create environment
RUN conda env create -f environment.yaml
# 3. use conda env properly in shell
SHELL ["conda", "run", "-n", "wordrec_swresnet", "/bin/bash", "-c"]
# 4. copy application
COPY . .
# Set Path
ENV PATH /opt/conda/envs/wordrec_swresnet/bin:$PATH
# 5. Environment
ENV PYTHONUNBUFFERED=1
# 6
ENV TORCH_HOME=/app/.cache/torch
# 6. Inference
CMD ["python", "swresnetx.py", "--model_name", "base_model"]