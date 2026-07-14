# AI Model Serving Platform

A hands-on MLOps project demonstrating the full pipeline for serving, tracking, and retrieving AI/ML workloads — built on top of an existing AWS EKS infrastructure.

## What This Project Covers

- **API Serving**: FastAPI application exposing model inference endpoints
- **Containerization**: Docker image built and optimized for deployment
- **Container Registry**: Image pushed to Amazon ECR
- **Kubernetes Deployment**: Deployed on AWS EKS with 2 replica pods behind a LoadBalancer service for high availability
- **Model Tracking**: MLflow used to log parameters, metrics, and model artifacts (trained a RandomForest classifier on the Iris dataset, achieving 100% test accuracy)
- **RAG Pipeline**: Retrieval-Augmented Generation pipeline using LangChain, HuggingFace sentence embeddings, and ChromaDB for semantic search over a document set

## Tech Stack

`FastAPI` `Docker` `Kubernetes (EKS)` `AWS ECR` `MLflow` `LangChain` `ChromaDB` `scikit-learn` `Python`

## Project Structure

- `main.py` — FastAPI application with serving endpoints
- `Dockerfile` — Container build definition
- `requirements.txt` — Python dependencies
- `train_experiment.py` — Basic MLflow experiment logging demo
- `train_real_model.py` — Real model training with MLflow tracking (RandomForest on Iris dataset)
- `rag_demo.py` — RAG retrieval pipeline demo

## What's Next

- KServe integration for Kubernetes-native model serving
- LLM-based answer generation to complete the RAG pipeline
- Production-grade vector database deployment
- Observability integration (Prometheus/Grafana) for AI workload metrics

## Background

Built as an extension of an existing DevOps/Cloud portfolio (AWS EKS, Terraform, CI/CD, Vault, Istio) to explore the infrastructure/deployment side of MLOps.