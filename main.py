from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()

# Загрузка модели YOLOv8
model = YOLO('best (2).pt')  # Замените на путь к вашей модели

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    print('yes')
    # Читаем изображение из файла
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Выполняем предсказание
    results = model(image)

    # Извлекаем названия классов
    if results and results[0].names:
        detected_classes = results[0].names
        predictions = results[0].pred[0]

        # Проверяем наличие предсказаний
        if len(predictions) > 0:
            class_id = int(predictions[0][5])  # ID класса
            fruit_name = detected_classes[class_id]
            return JSONResponse(content={"fruit": fruit_name})

    return JSONResponse(content={"error": "No fruit detected"})

@app.get("/")
async def main():
    return {"message": "Upload an image of a fruit to /predict/"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
