FROM condaforge/miniforge3:latest

WORKDIR /app

COPY environment-app.yml .
RUN mamba env create -f environment-app.yml

# activate the environment in the docker container adding the env path to system variables 
ENV PATH /opt/conda/envs/blood-donors-app/bin:$PATH

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]