## 快速啟動 VoxCPM2

1. 使用Docker建立環境:
   ```bash
   docker build -t voxcpm_api .
   ```

2. 啟動 API:
    ```bash
    docker run --rm —gpus all -d -p 10005:8000 --name tts_test2 voxcpm_api
    ```

    啟動後請訪問：http://<伺服器網址>:10005/docs 進入 Swagger UI 進行測試。

3. 關閉 API:
    ```bash
    docker stop tts_test2 
    docker rm tts_test2
    ```


## 使用說明

VoxCPM2的優勢為高音質，並同時支援音色克隆加上語調及情緒調整，可以用白話描述需要的聲音型態及場景

VoxCPM2共有三種功能：

1. Voice Design
    於text處輸入希望TTS模型生成的語句
    可於Prompt括號內()自訂說話者的語氣、情緒及性別。
    由於是中國研發的TTS模型，Prompt建議使用簡體中文以避免發音錯誤。

2. Voice Cloning
    於reference_wav_path載入希望克隆的音源
    （選配）可同時於括號內()自訂說話者的語氣及情緒

3. Ultimate Cloning
    高還原語音克隆
    於reference_wav_path及prompt_wav_path載入同一音源
    於prompt_text輸入音源的文字稿

4. 其他功能
    模型支援閩南語生成
    額外細節請參考原始模型[Github](https://github.com/OpenBMB/VoxCPM)
