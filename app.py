
from typing import Union
import torch
import transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Translation API", description="API for translating text using Sunbird NLLB model")

# Define request body model
class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

# Load the model and tokenizer (non-quantized for CPU compatibility)
model = transformers.M2M100ForConditionalGeneration.from_pretrained(
    "Sunbird/translate-nllb-1.3b-salt",
    torch_dtype=torch.float32
)
tokenizer = transformers.NllbTokenizer.from_pretrained("Sunbird/translate-nllb-1.3b-salt")

# Define language tokens
language_tokens = {
    'eng': 256047,
    'ach': 256111,
    'lgg': 256008,
    'lug': 256110,
    'nyn': 256002,
    'teo': 256006,
}

# Set device to CPU for cloud deployment
device = torch.device("cpu")
model = model.to(device)

@app.post("/translate", response_model=dict)
async def translate_text(request: TranslationRequest):
    try:
        # Validate language codes
        if request.source_language not in language_tokens or request.target_language not in language_tokens:
            raise HTTPException(status_code=400, detail="Invalid source or target language. Supported languages: " + ", ".join(language_tokens.keys()))

        # Tokenize input text
        inputs = tokenizer(request.text, return_tensors="pt").to(device)
        inputs['input_ids'][0][0] = language_tokens[request.source_language]

        # Generate translation
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=language_tokens[request.target_language],
            max_length=100,
            num_beams=5,
        )

        # Decode result
        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        return {
            "translated_text": result,
            "source_language": request.source_language,
            "target_language": request.target_language
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")
