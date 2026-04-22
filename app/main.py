from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from app.core import VoxCPMEngine
from contextlib import asynccontextmanager
import io, os, uuid, soundfile as sf
import torch
import gc


engine = VoxCPMEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("正在載入 VoxCPM 模型至 GPU...")
    engine.load_model()
    yield
    print("正在關閉 API 並釋放顯存...")
    if engine.model is not None:
        engine.model = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # 進階清理：清理進程間通訊的顯存

app = FastAPI(lifespan=lifespan)


# -------------------------
# Voice Design
# 於text處輸入希望TTS模型生成的語句
# 可於括號內()自訂說話者的語氣、情緒及性別。
# 由於是中國研發的TTS模型，Prompt建議使用簡體中文以避免發音錯誤。
# -------------------------

@app.post("/generate/voice_design")
async def voice_design(
    text: str = Form(
        ...,
        description="【必填】想要模型說出的文字內容，可於括號內()自訂說話者的語氣，使用簡體中文以避免發音錯誤。",
        example="(中年女性，温雅中性的声音)你好，我是一位虚拟助理，今天很高兴能够有这个机会认识各位，并和各位介绍功能。"
    ),
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
):
    wav, sr = engine.generate(
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps
        )
    return wav_to_stream(wav, sr)


# -------------------------
# Voice Cloning
# 於reference_wav_path載入希望克隆的音源
# （選配）可同時於括號內()自訂說話者的語氣及情緒
# -------------------------

@app.post("/generate/voice_cloning")
async def voice_cloning(
    text: str = Form(
        ...,
        description="【必填】想要模型說出的文字內容，可於括號內()自訂說話者的語氣，使用簡體中文以避免發音錯誤。",
        example="(中年女性，温雅中性的声音)你好，我是一位虚拟助理，今天很高兴能够有这个机会认识各位，并和各位介绍功能。"
    ),
    reference_wav_path: UploadFile = File(
        ...,
        description="【必填】參考音檔（樣本），15秒左右即可，可從Samples資料夾選取。",
    ),
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
):
    ref_path = await save_temp_file(reference_wav_path)

    try:
        wav, sr = engine.generate(
            text=text,
            reference_wav_path=ref_path,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps
        )
        return wav_to_stream(wav, sr)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(ref_path):
            os.remove(ref_path)
            print(f"已清理暫存檔: {ref_path}")


# -------------------------
# Ultimate Cloning
# 高還原語音克隆
# 於reference_wav_path及prompt_wav_path載入同一音源
# 於prompt_text輸入音源的文字稿
# -------------------------
@app.post("/generate/ultimate")
async def ultimate_cloning(
    text: str = Form(
        ...,
        description="【必填】想要模型說出的文字內容，高還原語音克隆不支援自訂說話者的語氣，使用簡體中文以避免發音錯誤。",
        example="你好，我是一位虚拟助理，今天很高兴能够有这个机会认识各位，并和各位介绍功能。"
    ),
    prompt_wav_path: UploadFile = File(
        ...,
        description="【必填】參考音檔（樣本），15秒左右即可，可從Samples資料夾選取。",
    ),
    prompt_text: str = Form(
        ...,
        description="【選填】上傳音檔的文字稿。",
    ),
    reference_wav_path: UploadFile = File(
        ...,
        description="【必填】參考音檔（樣本），和prompt_wav_path相同即可。",
    ),
):
    ref_path = await save_temp_file(reference_wav_path)

    try:
        wav, sr = engine.generate(
            text=text,
            prompt_wav_path=ref_path,
            prompt_text=prompt_text,
            reference_wav_path=ref_path
        )
        return wav_to_stream(wav, sr)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(ref_path):
            os.remove(ref_path)
            print(f"已清理暫存檔: {ref_path}")


async def save_temp_file(upload_file: UploadFile):
    ext = os.path.splitext(upload_file.filename)[1]
    tmp_path = f"temp_{uuid.uuid4()}{ext}"
    with open(tmp_path, "wb") as buffer:
        buffer.write(await upload_file.read())
    return tmp_path

def wav_to_stream(wav, sr):
    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format='WAV')
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")
