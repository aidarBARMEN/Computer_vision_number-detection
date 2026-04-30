from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from inference import CVModelManager
import uvicorn
import traceback

app = FastAPI(title="CV FastAPI — детекция + распознавание номеров")

model_manager = CVModelManager()


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    ocr_engine: str = Form("both"),
):
    try:
        print(f"📤 Запрос: модель={model_name} | ocr={ocr_engine} | файл={file.filename}")

        image_bytes = await file.read()
        prediction, original_b64, annotated_b64 = model_manager.predict(
            image_bytes, model_name, ocr_engine
        )

        return {
            "prediction": prediction,
            "original_image": original_b64,
            "annotated_image": annotated_b64,
            "model_used": model_name,
            "ocr_engine": ocr_engine,
        }

    except Exception as e:
        print("❌ Ошибка в /predict:")
        print(traceback.format_exc())
        return {"error": f"Ошибка сервера: {str(e)}"}, 500


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
