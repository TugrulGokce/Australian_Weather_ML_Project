FROM python:3.8

WORKDIR Australian_Weather_ML_Project/

COPY . /Australian_Weather_ML_Project

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]