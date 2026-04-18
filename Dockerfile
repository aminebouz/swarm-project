FROM python:3.11-slim

# Répertoire de travail
WORKDIR /app

# Dépendances système pour ChromaDB
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie et installe les dépendances d'abord (optimise le cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie le code
COPY simulation.py .

# Lance la simulation au démarrage
CMD ["python", "simulation.py"]
