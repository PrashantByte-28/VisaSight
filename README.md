# VisaSight — MLOps Pipeline for US Visa Approval Prediction

This is a machine learning project I built to predict whether a US work visa
petition is likely to get approved or denied, based on things like the
employee's education, job experience, wage, and the employer's company size.

I didn't want to just train a model in a notebook and leave it there, so I
built this as a full pipeline — data ingestion, validation, transformation,
training, evaluation — and then wrapped it in a FastAPI app so you can
actually fill out a form and get a live prediction. The app is deployed on
Vercel.

Dataset used: [EasyVisa dataset on Kaggle]


## 🌐 Live Demo

👉 <add your Vercel URL here after deploying>


## What it does

You fill out a short form (continent, education, experience, wage, company
size, etc.) and the model tells you whether it thinks the petition would be
approved or not.

During training, the pipeline tries HistGradientBoosting and RandomForest with
a grid search and keeps the best one. Wages are converted to a yearly amount
before modelling, because the dataset mixes hourly, weekly, monthly and yearly
values. On held-out test data the model gets about 70% accuracy, which is
reasonable for this dataset but means it is a demo, not something to rely on.

For monitoring, I added Evidently AI so I could check for data drift later
on, since that's something that gets skipped a lot in smaller projects but
matters in a real setup.

## 🐍 Python Version

The deployed app runs on **Python 3.12** (set in `.python-version`, which
Vercel reads).

The trained model is a pickle file, so it has to be loaded with the same
scikit-learn version it was trained with. That is why `scikit-learn` is pinned
in `requirements.txt`. If you retrain with a different version, update the pin
to match.

## Stack

- Python 3.12
- Scikit-learn (HistGradientBoosting, RandomForest)
- MongoDB (for storing the raw data, used only for training)
- FastAPI + Jinja2 for the web app
- Vercel for hosting

## How the pipeline is structured

I followed a fairly standard modular structure so each part can be tested
and run on its own:

```
constants -> entity -> components -> pipeline -> app.py
```

- **constants** – fixed config stuff (paths, DB names, thresholds)
- **entity** – config/artifact classes passed between stages
- **components** – ingestion, validation, transformation, training, evaluation
- **pipeline** – wires the components together (training pipeline + prediction pipeline)
- **app.py** – FastAPI app that loads the trained model and serves predictions

## Running it locally

```bash
git clone https://github.com/PrashantByte-28/VisaSight.git
cd VisaSight
# (repo renamed from Production-Ready-Machine-Learning-Project)

pip install -r requirements.txt
```

Run the app (uses the model saved in `saved_model/`):

```bash
python app.py
```

and go to `localhost:8080` in your browser.

### Training the model

Training needs a few extra packages that the deployed app doesn't use (so they
are not in `requirements.txt`), plus a MongoDB connection string:

```bash
pip install pymongo certifi evidently imbalanced-learn neuro-mf
export MONGODB_URL="mongodb+srv://<username>:<password>@..."
python demo.py
```

When training finishes, the model is saved to `saved_model/model.pkl`. Logs go
to `logs/` and intermediate files to `artifact/`.

## 🚀 Deployment

The app is deployed on **Vercel**. Vercel detects the FastAPI app in `app.py`
automatically, so no extra configuration file is needed.

```text
GitHub  →  Vercel  →  FastAPI app (loads saved_model/model.pkl)
```

- Training does **not** run on Vercel. I train locally, commit
  `saved_model/model.pkl`, and push.
- The deployed app needs no environment variables, no database and no cloud keys.
- Every push to `main` triggers a new deployment.

## Things I'd still like to improve

- Add proper unit tests for the pipeline components (currently none, which I know isn't great)
- Track experiments with MLflow instead of just comparing metrics manually
- Handle edge cases in the form better on the frontend
- Set up scheduled drift checks instead of only checking manually
- Try more feature engineering to push accuracy beyond about 70%

## Author

Prashant Kumar Mishra
[GitHub](https://github.com/PrashantByte-28) · [LinkedIn](https://www.linkedin.com/in/prashant-kumar-mishra-356004261)