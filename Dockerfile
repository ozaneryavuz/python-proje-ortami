FROM python:3.11-slim

RUN pip install --no-cache-dir pandas scikit-learn imbalanced-learn openpyxl

CMD ["python", "-m", "pip", "--version"]
