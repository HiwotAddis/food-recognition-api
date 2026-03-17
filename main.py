from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

app = FastAPI()

model = YOLO("best.pt")


def mask_to_rle(mask: np.ndarray) -> list[list[int]]:
    flat = mask.flatten()
    runs = []
    i = 0
    while i < len(flat):
        if flat[i] == 1:
            start = i
            while i < len(flat) and flat[i] == 1:
                i += 1
            runs.append([int(start), int(i - start)])
        else:
            i += 1
    return runs

@app.get("/")
def home():
    return {"message": "Food Recognition API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = model(image)

    detections = []

    for r in results:
        image_height, image_width = r.orig_shape

        for i, box in enumerate(r.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = model.names[class_id]

            # Bounding box in pixel coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection = {
                "food": name,
                "confidence": confidence,
                "box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            }

            # Include segmentation mask 
            if r.masks is not None:
                mask_xy = r.masks[i].xy[0].tolist()
                detection["mask_polygon"] = [
                    {"x": round(pt[0], 2), "y": round(pt[1], 2)}
                    for pt in mask_xy
                ]

                mask_np = r.masks.data[i].cpu().numpy().astype(np.uint8)
                mask_h, mask_w = mask_np.shape
                detection["mask_rle"] = mask_to_rle(mask_np)
                detection["mask_width"] = int(mask_w)
                detection["mask_height"] = int(mask_h)
                detection["pixel_area"] = int(mask_np.sum())

            detections.append(detection)

    return {
        "image_width": image_width,
        "image_height": image_height,
        "detections": detections
    }