#!make
include .env

install-requirements:
	pip install zenml

create_folders:
	if [ -d "minio_data" ]; then rm -rf minio_data; fi
	mkdir minio_data
	echo 'minio_data' folder created
	sleep 1

docker_up:
	docker compose up -d
	echo 'Docker containers started'

zenml-install-integrations:
	zenml integration install s3 -y
	zenml integration install sklearn -y
	zenml integration install mlflow -y

zenml-create-artifact-store-minio:
	zenml secret create minio_secret --aws_access_key_id=${AWS_ACCESS_KEY_ID} --aws_secret_access_key=${AWS_SECRET_ACCESS_KEY}
	zenml artifact-store register minio_store \
		--flavor=s3 \
		--path='s3://zenml' \
		--authentication_secret=minio_secret \
		--client_kwargs='{"endpoint_url": "http://localhost:9000", "region_name": "eu-west-2"}'

zenml-create-exp-tracker:	
	zenml secret create mlflow_secret \
		--username=${MLFLOW_TRACKING_USERNAME} \
		--password=${MLFLOW_TRACKING_PASSWORD} \
		--tracking_uri=${MLFLOW_TRACKING_URI} \
		--s3_endpoint_url=${MLFLOW_S3_ENDPOINT_URL}
	zenml experiment-tracker register mlflow_tracker \
		--flavor=mlflow \
		--tracking_username={{mlflow_secret.username}} \
		--tracking_password={{mlflow_secret.password}} \
		--tracking_uri={{mlflow_secret.tracking_uri}}

zenml-register-stack:
	zenml stack register ml-stack \
		--artifact-store=minio_store \
		--orchestrator=default \
		--experiment_tracker=mlflow_tracker

build:
	$(MAKE) install-requirements
	$(MAKE) create_folders
	$(MAKE) docker_up
	echo "Waiting for containers startup (30 seconds)"
	sleep 30
	zenml login http://localhost:8080 --refresh
	zenml login
	$(MAKE) zenml-install-integrations
	$(MAKE) zenml-create-artifact-store-minio
	$(MAKE) zenml-create-exp-tracker
	$(MAKE) zenml-register-stack
	zenml stack set ml-stack
	echo 'Project started'
