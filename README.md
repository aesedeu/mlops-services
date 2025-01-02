# mlops full stack project
Инструкция docker deploy
> https://docs.zenml.io/getting-started/deploying-zenml/deploy-with-docker#zenml-server-with-docker-compose

Еще какой-то мануал с готовым проектом (не работает):
> https://github.com/codecentric/from-jupyter-to-production-ml-platform

# Инструкция по сборке проекта
1. Заполняем файл с переменными окружения и переименовываем его в `.env`

2. Поднимаем проект
```bash
docker compose up -d
zenml login http://localhost:8080/ # логинимся через CLI
```
Логинимся через UI zenml (создаем администратора)

3. Устанавливаем необходимые зависимости
```bash
zenml integration install -y s3
source .env
zenml secret create minio_secret --aws_access_key_id=${AWS_ACCESS_KEY_ID} --aws_secret_access_key=${AWS_SECRET_ACCESS_KEY}
zenml artifact-store register minio_store -f s3 --path='s3://zenml' --authentication_secret=minio_secret --client_kwargs='{"endpoint_url": "http://localhost:9000", "region_name": "eu-east-1"}'

zenml integration install sklearn -y
```
4. Через UI zenml создаем новый стак:
- **Оркестратор:** python
- **Storage:** minio_store

5. Активируем новый стак “ml-stack”
```bash
zenml stack set ml-stack
zenml stack describe # должен отображен созданный стак
```

# Пример запуска проекта
```bash
python main.py
```