from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    return {"message: welcome to emotion recognition app"}