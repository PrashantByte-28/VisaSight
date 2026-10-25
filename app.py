import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from us_visa.constant import APP_HOST, APP_PORT
from us_visa.pipeline.prediction_pipeline import (
    USvisaData,
    USvisaClassifier,
)

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="VisaSight — US Visa Prediction",
    description="Machine learning-based US visa prediction demo.",
    version="1.0.0",
)

# Serve CSS and other static files using absolute paths.
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

# Load Jinja2 templates using an absolute path.
templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR),
)

# --------------------------------------------------
# CORS configuration
# --------------------------------------------------

# Configure allowed origins using a comma-separated
# ALLOWED_ORIGINS environment variable if required.
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False if "*" in allowed_origins else True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Form data handling
# --------------------------------------------------


class DataForm:
    def __init__(self, request: Request):
        self.request = request

        self.continent: Optional[str] = None
        self.education_of_employee: Optional[str] = None
        self.has_job_experience: Optional[str] = None
        self.requires_job_training: Optional[str] = None
        self.no_of_employees: Optional[str] = None
        self.company_age: Optional[str] = None
        self.region_of_employment: Optional[str] = None
        self.prevailing_wage: Optional[str] = None
        self.unit_of_wage: Optional[str] = None
        self.full_time_position: Optional[str] = None

    async def get_usvisa_data(self):
        form = await self.request.form()

        self.continent = form.get("continent")
        self.education_of_employee = form.get(
            "education_of_employee"
        )
        self.has_job_experience = form.get(
            "has_job_experience"
        )
        self.requires_job_training = form.get(
            "requires_job_training"
        )
        self.no_of_employees = form.get("no_of_employees")
        self.company_age = form.get("company_age")
        self.region_of_employment = form.get(
            "region_of_employment"
        )
        self.prevailing_wage = form.get("prevailing_wage")
        self.unit_of_wage = form.get("unit_of_wage")
        self.full_time_position = form.get(
            "full_time_position"
        )


# --------------------------------------------------
# Home page
# --------------------------------------------------


@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="usvisa.html",
        context={"context": "Rendering"},
    )


# --------------------------------------------------
# Training endpoint
# --------------------------------------------------


@app.get("/train", include_in_schema=False)
async def trainRouteClient():
    # Training is disabled by default on Vercel.
    if (
        os.getenv("ENABLE_TRAINING_ENDPOINT", "false").lower()
        != "true"
    ):
        return Response(
            content="Training endpoint is disabled.",
            status_code=404,
        )

    try:
        # Import the training pipeline only when explicitly enabled.
        from us_visa.pipeline.training_pipeline import TrainPipeline

        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        return Response(
            content="Training successful!",
            status_code=200,
        )

    except Exception:
        # Avoid exposing internal paths, credentials, or stack traces.
        print(
            "Training failed.",
            file=sys.stderr,
        )

        return Response(
            content="Training failed. Check application logs.",
            status_code=500,
        )


# --------------------------------------------------
# Visa prediction endpoint
# --------------------------------------------------


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        # Read submitted form data.
        form = DataForm(request)
        await form.get_usvisa_data()

        # Create the prediction input.
        usvisa_data = USvisaData(
            continent=form.continent,
            education_of_employee=form.education_of_employee,
            has_job_experience=form.has_job_experience,
            requires_job_training=form.requires_job_training,
            no_of_employees=form.no_of_employees,
            company_age=form.company_age,
            region_of_employment=form.region_of_employment,
            prevailing_wage=form.prevailing_wage,
            unit_of_wage=form.unit_of_wage,
            full_time_position=form.full_time_position,
        )

        # Convert the input into a DataFrame.
        usvisa_df = usvisa_data.get_usvisa_input_data_frame()

        # Load the bundled trained model (saved_model/) and
        # predict the result.
        model_predictor = USvisaClassifier()
        prediction = model_predictor.predict(
            dataframe=usvisa_df
        )[0]

        # Preserve the existing result labels.
        status = (
            "Visa Not-Approved"
            if prediction == 1
            else "Visa-approved"
        )

        # Render the existing result page.
        return templates.TemplateResponse(
            request=request,
            name="usvisa.html",
            context={"context": status},
        )

    except Exception:
        # Log the exception server-side without exposing internals.
        import logging

        logging.exception("Visa prediction failed.")

        return templates.TemplateResponse(
            request=request,
            name="usvisa.html",
            context={
                "context": "Prediction could not be completed. "
                "Please try again later."
            },
            status_code=500,
        )


# --------------------------------------------------
# Local development
# --------------------------------------------------

if __name__ == "__main__":
    app_run(
        app,
        host=APP_HOST,
        port=APP_PORT,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )