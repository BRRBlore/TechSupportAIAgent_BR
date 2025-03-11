 
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = FastAPI()

# ✅ Correct Model Path for Google Drive
model_path = "/content/drive/MyDrive/AI_Agents_BR/trained_models_sales_only_continued/checkpoint-11775"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

class Query(BaseModel):
    customer_message: str

@app.post("/generate_response/")
async def generate_response(query: Query):
    input_ids = tokenizer.encode(
        query.customer_message,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    ).to(model.device)

    outputs = model.generate(
        input_ids, 
        max_length=256, 
        num_beams=4, 
        early_stopping=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"agent_response": response}
