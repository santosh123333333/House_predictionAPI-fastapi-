# 🏠 House Price Prediction API

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)


A production-ready Machine Learning API built with **FastAPI** that predicts house prices based on property features. This project demonstrates the complete lifecycle of an ML application: from data exploration and model training to deploying a fast, interactive RESTful API.

---

## 📖 Table of Contents
1. [Features](#-features)
2. [Project Structure](#-project-structure)
3. [Installation & Setup](#%EF%B8%8F-installation--setup)
4. [Interactive API Documentation (Swagger UI)](#-interactive-api-documentation-swagger-ui)
5. [API Endpoints](#-api-endpoints)
6. [Model Training](#-model-training)
7. [Technologies Used](#-technologies-used)

---

## 🚀 Features

- **Machine Learning Model:** Trained regression model saved as `.joblib` for fast, lightweight inference.
- **FastAPI Framework:** High-performance, asynchronous web framework.
- **Data Validation:** Uses **Pydantic** to ensure incoming data is correctly formatted before prediction.
- **Interactive API Docs:** Auto-generated, fully interactive Swagger UI (no extra configuration needed).
- **Clean Architecture:** Separated concerns between model training (`train.py`) and API serving (`main.py`).

---

## 📂 Project Structure

```text
fastapimlProject/
│
├── app/                        # Main application folder
│   ├── __pycache__/            # Python bytecode (ignored by git)
│   ├── house_feature.joblib    # Saved feature list/transformers
│   ├── house_model.joblib      # Trained ML model
│   ├── main.py                 # FastAPI application & endpoints
│   └── train.py                # Script to train the model
│
├── explore/                    # Data exploration scripts
│   └── explore.py              # EDA (Exploratory Data Analysis) code
│
├── model/                      # Directory for model artifacts
│
├── .gitignore                  # Files ignored by Git (.venv, *.joblib, etc.)
├── req.txt                     # Python dependencies
└── README.md                   # Project documentation

## 📖 Interactive API Documentation (Swagger UI)

FastAPI automatically generates interactive documentation. Once the server is running, open your browser and navigate to:

👉 **`http://127.0.0.1:8000/docs`**

Here you can test the `/predict` endpoint directly from the browser by entering house features.

![Swagger UI Interface](images/swagger.png)


