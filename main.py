from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from Detector import Detector
import logging

app = FastAPI()

logger = logging.getLogger("mainLogger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('custom.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    product_detector = Detector(contents)
    result_data = product_detector.detect()
    logger.info("Detection completed")
    return JSONResponse(content=result_data)

@app.get("/")
async def main():
    return {"message": "Upload an image of a fruit to /predict/"}

@app.on_event("startup")
def startup_event():
    logger.info("Server started")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Server stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
