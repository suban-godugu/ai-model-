from fastapi import FastAPI, UploadFile, File
import shutil
import os

from src.predict import predict_api

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(

    title="WaferVision AI API",

    version="1.0"
)

# =====================================================
# TEMP FOLDER
# =====================================================

UPLOAD_DIR = "temp"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

# =====================================================
# HOME ROUTE
# =====================================================

@app.get("/")
def home():

    return {

        "message": "WaferVision AI API Running"
    }

# =====================================================
# PREDICTION ROUTE
# =====================================================

@app.post("/predict")
async def predict(

    file: UploadFile = File(...)
):

    # =============================================
    # SAVE IMAGE
    # =============================================

    file_path = os.path.join(

        UPLOAD_DIR,

        file.filename
    )

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(

            file.file,

            buffer
        )

    # =============================================
    # RUN MODEL
    # =============================================

    result = predict_api(
        file_path
    )

    # =============================================
    # RETURN JSON
    # =============================================

    return result