# Use uma imagem base do Python
FROM python:3.10-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia os requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia apenas a pasta src
COPY src/ ./src/

# Expõe a porta que a aplicação usa
EXPOSE 5001

# Comando para executar a aplicação
CMD ["python", "src/app.py"]