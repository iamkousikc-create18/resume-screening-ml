# 📄 Resume Screening using Machine Learning

A machine learning-powered web application that automatically classifies resumes into different job categories based on their content. This project uses **TF-IDF Vectorization** and a **Random Forest Classifier** and is deployed with **Streamlit**.

---

## 🚀 Live Demo

👉 Add your Streamlit deployment link here

---

## 📌 Overview

Recruiters often receive hundreds of resumes for various job roles. Manually screening them is time-consuming and inefficient.

This project automates the resume screening process by predicting the most suitable job category from a resume using Natural Language Processing (NLP) and Machine Learning.

---

## ✨ Features

- 📄 Paste resume text directly
- 📂 Upload resume in **PDF** format
- 📃 Upload resume in **TXT** format
- 🧹 Resume text cleaning and preprocessing
- 🤖 Job category prediction
- 📊 Prediction confidence score
- 📈 Interactive probability chart
- 🎨 User-friendly Streamlit interface

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- PyPDF2
- Pickle

### Machine Learning
- TF-IDF Vectorizer
- Random Forest Classifier

---

## 📂 Project Structure

```
resume-screening-ml/
│
├── app.py
├── Resume Screening with Python.ipynb
├── UpdatedResumeDataSet.csv
├── rf_model.pkl
├── tfidf.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Dataset used:

**UpdatedResumeDataSet.csv**

The dataset contains resumes belonging to multiple job categories such as:

- Data Science
- Python Developer
- Java Developer
- HR
- Testing
- DevOps Engineer
- Blockchain
- Mechanical Engineer
- Civil Engineer
- Business Analyst
- Sales
- Advocate
- Health and Fitness
- Electrical Engineering
- Operations Manager
- SAP Developer
- ETL Developer
- Web Designing
- PMO
- Database
- Network Security Engineer
- Arts
- And more...

---

## ⚙️ Machine Learning Workflow

```
Resume
      ↓
Text Cleaning
      ↓
TF-IDF Vectorization
      ↓
Random Forest Classifier
      ↓
Predicted Job Category
```

---

## 📈 Model Performance

The model was trained using:

- TF-IDF Vectorizer
- Random Forest Classifier

Evaluation was performed using:

- Accuracy Score
- Classification Report
- Confusion Matrix

---

## 💻 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/resume-screening-ml.git
```

```bash
cd resume-screening-ml
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 📷 Application Preview

You can add screenshots here after deployment.

Example:

```
Home Page Screenshot

Prediction Result Screenshot
```

---

## 🔮 Future Improvements

- Support DOCX resumes
- Resume ranking system
- Skill extraction
- Resume summary generation
- ATS compatibility score
- Resume recommendations
- Deploy with Docker

---

## 👨‍💻 Author

**Kousik Chakraborty**

B.Tech in Data Science Engineering

---
