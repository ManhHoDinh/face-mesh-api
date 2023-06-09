from typing import Annotated
from fastapi import FastAPI, File, UploadFile
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World 2"}

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}