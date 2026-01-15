import asyncio
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= CONFIG =================
# Model HY-MT1.5-1.8B đạt 90% hiệu năng Gemini-3.0-Pro [cite: 21]
MODEL_NAME = "tencent/HY-MT1.5-1.8B" 
# MODEL_NAME = "tencent/HY-MT1.5-7B" 
DEVICE = "cuda:0"
BATCH_SIZE = 32          # Tối ưu cho RTX 4090 
BATCH_TIMEOUT = 0.05     # 50ms để gom đủ batch
MAX_NEW_TOKENS = 2048

# ================= LOAD MODEL =================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Cần thiết cho batch inference
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

print("Loading model...")
# Sử dụng bfloat16 để tối ưu độ chính xác trên kiến trúc RTX 40-series
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16, 
    device_map={"": DEVICE},
    trust_remote_code=True
).eval()

# ================= FASTAPI =================
app = FastAPI(title="HY-MT1.5 Optimized API")

class TranslationReq(BaseModel):
    text: str
    tgt: str
    context: Optional[str] = None
    terminology: Optional[Dict[str, str]] = None

request_queue: asyncio.Queue = asyncio.Queue()

def format_messages(req: TranslationReq) -> List[Dict[str, str]]:
    """Xây dựng nội dung tin nhắn theo hướng dẫn của HY-MT [cite: 304, 307]"""
    system_content = "You are a professional translator."
    user_content = ""
    
    # Tích hợp ngữ cảnh (Context) [cite: 149, 311]
    if req.context:
        user_content += f"Context: {req.context}\n"
    
    # Tích hợp thuật ngữ (Terminology) [cite: 149, 308]
    if req.terminology:
        terms = ", ".join([f"'{k}' translate to '{v}'" for k, v in req.terminology.items()])
        user_content += f"Terminology: {terms}\n"

    # Lệnh dịch chuẩn [cite: 143, 304]
    user_content += f"Translate the following text into {req.tgt}. Output ONLY the translation.\nText: {req.text}"
    
    return [
        {"role": "user", "content": user_content}
    ]

# ================= ENDPOINT =================
@app.post("/translate")
async def translate(req: TranslationReq):
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    await request_queue.put((req, future))
    return await future

# ================= GPU WORKER =================
async def gpu_worker():
    print(f"GPU worker started on {DEVICE}")
    while True:
        batch = []
        futures = []

        try:
            # Chờ gom request
            while len(batch) < BATCH_SIZE:
                req_data, fut = await asyncio.wait_for(
                    request_queue.get(), timeout=BATCH_TIMEOUT
                )
                batch.append(req_data)
                futures.append(fut)
        except asyncio.TimeoutError:
            pass

        if not batch:
            continue

        # Chuẩn bị input theo format của HY-MT
        prompts = [
            tokenizer.apply_chat_template(
                format_messages(r), 
                tokenize=False, 
                add_generation_prompt=True
            ) for r in batch
        ]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        
        input_seq_len = inputs.input_ids.shape[1]
        
        # Công thức: (Độ dài đầu vào * hệ số giãn nở) + hạn mức bù trừ
        # Chúng ta giới hạn tối thiểu 32 và tối đa MAX_NEW_TOKENS để an toàn
        dynamic_max_tokens = min(MAX_NEW_TOKENS, max(32, int(input_seq_len * 1.5) + 20))

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=dynamic_max_tokens,
                do_sample=False,
                # Tăng penalty để tránh lặp lại 
                repetition_penalty=1.2, 
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                forced_eos_token_id=tokenizer.eos_token_id 
            )
        # Chỉ lấy phần text mới được generate
        input_len = inputs.input_ids.shape[1]
        decoded = tokenizer.batch_decode(
            outputs[:, input_len:], 
            skip_special_tokens=True
        )

        for fut, result in zip(futures, decoded):
            print(f"Translation result: {result}")
            fut.set_result({
                "translation": result.strip(),
                "tokens_generated": dynamic_max_tokens,
                "model": "HY-MT1.5-1.8B"
            })

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(gpu_worker())