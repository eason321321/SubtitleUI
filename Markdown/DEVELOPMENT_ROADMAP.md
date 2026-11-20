# SubtitleUI 開發地圖（Development Roadmap）

## 1. 專案背景與目標

**SubtitleUI** 是一個輕量化、本機運行的韓文影片語音辨識與學習工具，初期以 Windows 為主要開發與使用環境。

使用者只需提供網路影片網址，系統即可在本機下載影片或音訊，並於後續階段整合本機開源 ASR（Automatic Speech Recognition）模型，將韓文語音轉換為具備時間戳記的逐字稿，最終打造出整合影片播放、逐字稿同步與韓文輔助學習功能的桌面應用程式。

---

## 2. 核心設計原則

1. **Local-first**：優先採用本機運算與本機檔案管理，不依賴付費雲端 API。
2. **Lightweight**：精簡依賴，避免不必要的框架、背景服務或過度膨脹的套件。
3. **Modular**：將下載、辨識、逐字稿處理、匯出、介面等職責拆分為獨立模組，確保核心邏輯與 UI 解耦，方便後續替換模型或擴充功能。
4. **Incremental Development**：嚴格遵循分階段循序開發，僅實作當前階段範圍內的功能，不提前堆砌未排定需求的程式碼。
5. **Maintainability**：保持程式碼結構清晰、介面明確、錯誤處理完善，避免過度工程化。

---

## 3. 技術架構

整體採分層與模組化設計，將核心業務邏輯（下載、ASR、逐字稿處理）與操作介面（初期 CLI、後續 PySide6 GUI）分離：

```text
[ 使用者介面層 ]
  Phase 1~3: CLI 入口 (main.py / cli)
  Phase 4~6: 桌面 GUI (PySide6)
        │
        ▼
[ 核心功能模組層 (src/subtitleui/) ]
  ├── downloader/   (Phase 1) 影片/音訊下載、進度回報、錯誤封裝
  ├── asr/          (Phase 2) 本機語音辨識 (Faster-Whisper)
  ├── transcript/   (Phase 3) 斷句、時間戳整理、SRT/TXT/JSON 匯出
  └── learning/     (Phase 6) 單字查詢、翻譯、收藏與學習紀錄
        │
        ▼
[ 本機外部工具與資料層 ]
  ├── yt-dlp / FFmpeg (媒體取得與音訊轉檔)
  ├── downloads/      (下載之媒體檔案，不納入 Git)
  └── models/         (本機 ASR 模型權重，不納入 Git)
```

---

## 4. 使用的主要套件及其用途

| 套件 / 工具 | 導入階段 | 用途說明 |
| :--- | :---: | :--- |
| **Python 3.10+** | Phase 1 | 專案主要開發語言與標準函式庫支援 |
| **yt-dlp** | Phase 1 | 解析網路影片網址並串流下載影片或音訊至本機 |
| **FFmpeg** | Phase 1 / 2 | 影音串流合併（Phase 1）與音訊擷取／格式轉換（Phase 1、2） |
| **pytest**（開發測試） | Phase 1 | 單元測試與 Mock 測試執行（亦相容標準庫 `unittest`） |
| **Faster-Whisper / CTranslate2** | Phase 2 | 基於 CTranslate2 的高效本機 Whisper 語音辨識引擎（支援 CUDA FP16 / CPU INT8） |
| **nvidia-cublas-cu12 / nvidia-cudnn-cu12**（選用） | Phase 2 | Windows NVIDIA GPU 加速所需之 CUDA 12 / cuDNN 9 執行期函式庫（未安裝時自動降級 CPU INT8） |
| **PySide6** | Phase 4 | 跨平台桌面 GUI 框架（Qt for Python） |

> 註：專案遵循最小依賴原則，每個階段僅安裝該階段實際需要的套件。

---

## 5. 各階段開發目標與完成標準

### Phase 1：影片網址輸入與下載
- **目標**：建立最基本的影片與音訊取得功能，並封裝成可重用的獨立下載模組與輕量 CLI 入口。
- **核心功能**：
  1. 接受使用者輸入的影片網址。
  2. 整合 `yt-dlp`。
  3. 支援下載影片（video）及音訊（audio）。
  4. 支援指定下載模式。
  5. 將下載內容儲存至本機指定資料夾（不存在時自動建立）。
  6. 即時顯示下載進度與處理狀態。
  7. 妥善處理無效網址、網路錯誤、下載中斷／失敗、同名檔案衝突等異常情境。
- **完成標準**：
  - 能成功輸入影片網址。
  - 能成功下載影片或音訊（串流寫入磁碟，不將整個檔案載入記憶體）。
  - 能正確儲存檔案至指定輸出資料夾。
  - 能妥善處理常見錯誤，且不會因下載失敗而導致整個程式崩潰。
  - 具備完整自動化測試（以 Mock 為主，涵蓋主要正常與異常情境）。

### Phase 2：本機語音辨識
- **目標**：將本機影片或音訊轉換成帶有原始影片時間戳記的韓文逐字稿，並支援整部影片與自訂時間範圍辨識。
- **核心技術**：Faster-Whisper、CTranslate2、Whisper 模型、FFmpeg / FFprobe、JSON。
- **核心功能**：
  1. 載入並快取本機 Faster-Whisper 模型（`TranscriptionEngine` 可重複使用，不重複初始化）。
  2. 支援韓文（`ko`）語音辨識，預設任務為 `transcribe`，保留模型原始輸出不自動翻譯或改寫。
  3. 支援依設定／CLI 參數切換模型（`small`、`medium`、`large-v3`、`large-v3-turbo`，初期預設 `medium`）。
  4. 支援整部影片（模式 A）與自訂時間範圍（模式 B，`HH:MM:SS` 驗證與 FFmpeg 精確擷取 16 kHz 單聲道 PCM WAV，不使用 stream copy）。
  5. 自動執行時間戳記偏移校正（`original_timestamp = segment_timestamp + selected_start_time`），確保輸出對齊原影片時間軸。
  6. 硬體策略自動切換：優先 `CUDA + FP16`，GPU 或執行期函式庫不可用／初始化失敗時自動備援 `CPU + INT8`。
  7. 儲存原始辨識結果為統一 UTF-8 JSON 格式（含中繼資料與模型回報之語言信心資訊），並支援工作階段回報與取消機制（`CancellationToken`）。
- **完成標準**：
  - 能在本機載入指定 Whisper 模型並完成韓文音訊轉錄。
  - 能指定原影片開始與結束時間，僅對擷取區間進行辨識並輸出正確偏移後的原影片時間戳記。
  - 能輸出具時間戳記的結構化原始辨識結果 JSON 檔案。
  - 能妥善處理缺少模型、無效時間範圍、音訊不存在、空音訊、取消操作與 GPU 不可用備援。

### Phase 3：逐字稿處理與匯出
- **目標**：將原始 ASR 結果整理為易於閱讀、編輯與外掛使用的逐字稿格式。
- **核心功能**：
  1. 句子切分與標點／斷句整理。
  2. 時間戳記校正與格式化。
  3. SRT 字幕檔匯出。
  4. TXT 純文字檔匯出。
  5. JSON 結構化資料匯出。
  6. 完整保留原始辨識結果不被覆寫。
- **完成標準**：
  - 能從原始辨識結果穩定轉換並匯出合法的 SRT、TXT 與 JSON 檔案。
  - 原始辨識檔案維持完整保留。

### Phase 4：桌面 GUI
- **目標**：建立簡潔直覺的桌面應用程式，整合 Phase 1~3 核心模組。
- **核心技術**：Python + PySide6。
- **核心功能**：
  1. 影片網址輸入與下載模式選擇。
  2. 下載進度與狀態顯示（非同步執行，不阻塞介面）。
  3. 非同步取消下載機制（Cancellation Token）：擴充 `downloader` 模組，於 `progress_hook` 加入可外部觸發的取消旗標（如 `threading.Event` / Cancellation Token），在觸發取消時安全中止 `yt-dlp` 下載並拋出 `DownloadInterruptedError`，支援 GUI「取消下載」按鈕。
  4. ASR 模型選擇。
  5. 語音辨識進度顯示。
  6. 逐字稿瀏覽區顯示。
  7. 一鍵匯出功能（SRT / TXT / JSON）。
- **完成標準**：
  - GUI 可直接呼叫既有核心模組完成「下載 → 辨識 → 瀏覽 → 匯出」完整流程，且操作過程介面流暢不卡死。
  - 使用者點擊「取消下載」按鈕時，能透過 Cancellation Token 安全中止背景下載任務並回復介面狀態，不造成程式崩潰。

### Phase 5：影片與逐字稿同步
- **目標**：建立真正可用的韓文聽力學習播放介面。
- **核心功能**：
  1. 內建影片／音訊播放器。
  2. 逐字稿與影片播放時間即時同步。
  3. 點擊逐字稿句子跳轉至指定時間點。
  4. 自動高亮目前正在播放的句子。
  5. 單句循環播放（A-B Repeat / Sentence Loop）。
  6. 播放速度調整（如 0.5x ~ 2.0x）。
- **完成標準**：
  - 影片播放與逐字稿同步誤差在可接受範圍內，點擊跳轉、單句循環與倍速播放皆運作穩定。

### Phase 6：韓文學習功能
- **目標**：將工具進一步擴充為個人化韓文聽力與詞彙學習環境。
- **核心功能**：
  1. 單字查詢（字典／詞性／原型分析）。
  2. 韓文中文翻譯。
  3. 生詞收藏。
  4. 句子收藏。
  5. AI 語境解釋（支援本機或可選外部模型介面）。
  6. 單字及句子學習紀錄與複習管理。
- **完成標準**：
  - 使用者可在閱讀逐字稿過程中即時查詞、翻譯、收藏並留存本機學習紀錄。

---

## 6. 專案目錄規劃

採用 `src/` layout，維持模組邊界清晰且易於測試與打包。下列規劃涵蓋目前（Phase 1~2）與未來預留位置（實際開發時僅建立當階段所需檔案）：

```text
SubtitleUI/
├── Markdown/
│   └── DEVELOPMENT_ROADMAP.md   # 專案開發地圖與進度紀錄（保留目錄既有大小寫）
├── src/
│   └── subtitleui/
│       ├── __init__.py
│       ├── __main__.py          # (Phase 1) 支援 `python -m subtitleui`
│       ├── cli.py               # (Phase 1~2) 輕量 CLI 入口（下載與本機 ASR 辨識）
│       ├── downloader/          # (Phase 1) 影片與音訊下載模組
│       │   ├── __init__.py
│       │   ├── errors.py        # 自訂錯誤類型階層
│       │   ├── models.py        # 下載請求、進度與結果資料結構
│       │   ├── validators.py    # 網址、音訊格式與輸出目錄驗證
│       │   └── service.py       # yt-dlp Python API 下載核心實作
│       ├── asr/                 # (Phase 2) 本機 Faster-Whisper 語音辨識模組
│       │   ├── __init__.py
│       │   ├── errors.py        # ASR 自訂錯誤類型階層
│       │   ├── models.py        # 設定、進度、片段、結果與 CancellationToken
│       │   ├── time_utils.py    # 時間格式解析、區間驗證與時間戳記偏移
│       │   ├── audio.py         # FFprobe 探測與 FFmpeg 16kHz 單聲道 WAV 擷取
│       │   ├── hardware.py      # Windows NVIDIA DLL 註冊與 CUDA/CPU 備援策略
│       │   ├── storage.py       # 統一 UTF-8 JSON 逐字稿讀寫
│       │   └── engine.py        # 可重用 TranscriptionEngine 辨識核心
│       ├── transcript/          # (Phase 3 預留，現階段不建立)
│       ├── ui/                  # (Phase 4~5 預留，現階段不建立)
│       └── learning/            # (Phase 6 預留，現階段不建立)
├── tests/
│   ├── __init__.py
│   ├── test_downloader.py       # (Phase 1) 下載模組與驗證單元測試
│   ├── test_asr.py              # (Phase 2) ASR 引擎、時間偏移、備援、取消與 FFmpeg 測試
│   └── test_cli.py              # (Phase 1~2) CLI 下載與辨識流程測試
├── downloads/                   # 預設下載輸出資料夾（不納入 Git）
├── models/                      # 本機 Faster-Whisper 模型快取資料夾（不納入 Git）
├── main.py                      # (Phase 1~2) 根目錄快速啟動腳本
├── .gitignore
└── requirements.txt             # 執行期核心依賴（yt-dlp、faster-whisper）
```

---

## 7. 未來擴充方向

- **批次處理與播放清單支援**：一次排程下載並辨識多個影片。
- **多語種擴充**：在穩定支援韓文後，開放切換日文、英文等 Whisper 支援語系。
- **雙語字幕與匯出**：支援韓中雙語 SRT 匯出與 Anki 單字卡匯出。
- **硬體加速自動偵測**：在 Phase 2 自動判斷 CUDA GPU 與 CPU（INT8 / FP16）最佳推論設定。
- **本機學習資料匯出／備份**：提供純文字或 JSON 備份生詞本與學習紀錄。

---

## 8. 開發限制與注意事項

1. **嚴格分階段開發**：未經規劃與進入下一階段前，不得提前撰寫後續階段的程式碼或引入其依賴。
2. **版本控制衛生**：禁止將大型影片、音訊檔、模型權重檔、虛擬環境（`.venv/`）、快取檔或任何憑證／Token 提交至 Git。
3. **記憶體與資源控制**：媒體下載與處理應採串流寫入磁碟的方式，避免將整部影片載入記憶體。
4. **合法與平台相容性**：`yt-dlp` 隨外部網站變動可能需要定期更新；專案僅供個人學習用途。
5. **Windows 路徑與編碼相容**：需注意 Windows 檔案名稱非法字元處理、長路徑、UTF-8 輸出編碼以及 FFmpeg 偵測提示。

---

## 9. 開發進度與階段紀錄

### Phase 1 狀態
- **目前狀態**：已完成（Completed）
- **完成日期**：2026-11-20
- **環境與依賴版本**：
  - Python `3.10.11`（透過 Windows `py` launcher 建立 `.venv`）
  - `yt-dlp` `2026.10.21`
  - FFmpeg `9.0-full_build`（系統 PATH 可用）
  - 測試工具：Python 標準庫 `unittest`（零額外測試套件依賴）
- **實作摘要**：
  - 建立獨立下載模組 `src/subtitleui/downloader/`（`models.py`、`errors.py`、`validators.py`、`service.py`），以 `yt-dlp` Python API 串流寫入磁碟，不將整個媒體檔載入記憶體。
  - 支援 `video`（有 FFmpeg 時合併為 `mp4`，無 FFmpeg 時自動降級單一串流）與 `audio`（透過 FFmpeg 提取音訊，預設 `m4a`，可指定 `mp3`/`wav` 等格式）兩種模式。
  - 支援自動建立不存在的輸出目錄、保留韓文檔名並移除 Windows 非法字元（`windowsfilenames=True`）。
  - 支援同名檔案衝突策略：`skip`（預設略過並回傳既有檔案）、`overwrite`（覆寫）、`fail`（拋出 `FileAlreadyExistsError`）。
  - 提供結構化 `DownloadProgress` callback 機制，可供 CLI 即時顯示進度，亦可於 Phase 4 直接銜接 PySide6 GUI。
  - 建立簡潔 CLI 入口（`main.py` / `src/subtitleui/cli.py`），支援命令列參數與互動式輸入，所有下載錯誤與中斷皆安全攔截，不造成程式崩潰。
- **測試結果**：
  - 執行 `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`：**20 / 20 項測試全部通過（0.170s）**。
  - 測試涵蓋項目：
    1. 有效影片網址下載（含韓文標題、進度回報序列、中繼資料與檔案大小驗證）。
    2. 無效網址驗證（空白、缺協定、不支援協定、畸形主機、包含空白、extractor 回報不支援網址）。
    3. 網路連線失敗映射（DNS 解析失敗、逾時、`URLError`、`TimeoutError` → `NetworkDownloadError`）。
    4. 下載中斷或失敗（`KeyboardInterrupt`、傳輸中斷、`PostProcessingError`、缺少 FFmpeg、影片不可用、無效音訊格式、下載結束後檔案遺失）。
    5. 輸出目錄不存在自動建立，以及輸出路徑指向一般檔案時報錯。
    6. 已存在同名檔案的三種策略（`skip` / `overwrite` / `fail`）。
    7. 影片（有／無 FFmpeg）與音訊兩種下載模式及真實 `yt_dlp.YoutubeDL` 選項相容性煙霧測試。
    8. CLI 命令列模式、互動模式、錯誤處理與中斷結束碼驗證。
- **尚未解決問題與注意事項**：
  1. 本階段自動化測試採 Mock（`FakeYoutubeDL`）與 `yt_dlp.YoutubeDL` 選項煙霧測試，尚未納入需要真實外部網路連線的端對端影片下載測試（因外部平台頁面與限制隨時可能變動，如 YouTube 需登入或年齡限制影片）。
  2. Phase 1 目前僅透過 `KeyboardInterrupt` 處理 CLI 中斷，尚未實作外部取消旗標；**已納入 Phase 4 正式開發項目**，將於 Phase 4 開發桌面 GUI 時於 `progress_hook` 補充可外部觸發的非同步取消機制（Cancellation Token），以支援介面「取消下載」按鈕。
  3. 目標網址為播放清單時目前預設 `noplaylist=True` 僅下載單支影片，批次下載將於後續階段視需求擴充。
  4. 本機 Windows 系統 PATH 中 `python` 指向 Microsoft Store alias，開發與啟動時需使用 `.venv\Scripts\python.exe` 或 `py` launcher。

### Phase 2 狀態
- **目前狀態**：已完成（Completed）
- **完成日期**：2026-11-20
- **環境與依賴版本**：
  - Python `3.10.11`（`.venv`）
  - `faster-whisper` `1.2.1`
  - `ctranslate2` `4.9.0`
  - `nvidia-cublas-cu12` `12.9.2.10`、`nvidia-cudnn-cu12` `9.28.1.4`、`nvidia-cuda-nvrtc-cu12` `12.9.86`（Windows GPU 執行期加速）
  - FFmpeg / FFprobe `9.0-full_build`
  - 本機實測硬體：NVIDIA GeForce GTX 1660 Ti（6 GB VRAM，Driver `591.74` / CUDA `13.1`）、16 GB RAM、Windows
- **實作摘要**：
  - 建立獨立本機 ASR 模組 `src/subtitleui/asr/`（`errors.py`、`models.py`、`time_utils.py`、`audio.py`、`hardware.py`、`storage.py`、`engine.py`），將模型載入、音訊擷取、ASR 推論與 JSON 持久化職責完全分離，方便 Phase 4 GUI 直接非同步呼叫。
  - 支援整部影片（模式 A）與自訂時間範圍（模式 B），提供 `HH:MM:SS` / `MM:SS` / 秒數解析與邊界驗證（負數、格式錯誤、起訖相同、結束小於開始、超出影片總長度）。
  - `AudioExtractor` 以 `ffprobe` 驗證音訊軌與總長度，並以 `ffmpeg` 精確解碼（`-ss` 置於 `-i` 後，不使用 stream copy）擷取 16 kHz 單聲道 `pcm_s16le` WAV 至暫存目錄，驗證實際長度並於辨識結束／取消／錯誤後自動清理暫存檔。
  - `TranscriptionEngine` 可重複使用已載入之 Faster-Whisper 模型（不重複初始化），正確迭代 `segments` generator，完整執行時間戳記偏移公式 `original_timestamp = segment_timestamp + selected_start_time`，保留原始韓文輸出不翻譯改寫，僅在模型有回報時記錄 `language_probability`。
  - 建立執行期硬體策略與 `register_nvidia_dll_directories()`，於 `auto` 模式優先採用 `CUDA + FP16`，若無 GPU、缺少 CUDA/cuDNN DLL 或初始化／首次推論發生 CUDA 錯誤時自動切換至 `CPU + INT8`。
  - 支援合作式取消機制（`CancellationToken` + `KeyboardInterrupt`）與非阻塞式階段狀態回報（顯示模型、運算裝置、處理階段、已處理音訊時間與實際耗時，不偽造百分比進度）。
  - 擴充 CLI（`src/subtitleui/cli.py`）新增 `transcribe` 子命令（支援命令列參數與互動模式），同時完全相容 Phase 1 下載指令。
- **模型設定與相容性**：
  - 支援常用模型：`small`、`medium`（預設）、`large-v3`、`large-v3-turbo`（並支援本機模型資料夾路徑），可透過 `--model` 或 `ModelConfig` 動態指定，未在程式碼寫死。
  - 模型快取於專案根目錄 `models/`（已加入 `.gitignore`），首次下載顯示清楚提示，後續直接讀取本機快取；並設定 `HF_HUB_DISABLE_SYMLINKS_WARNING=1` 避免 Windows 非開發者模式出現大量警告。
  - **GTX 1660 Ti（6 GB VRAM）相容性**：`ctranslate2 4.9.0` 需搭配 CUDA 12（`cublas64_12.dll`）與 cuDNN 9（`cudnn_ops64_9.dll`）；透過 `.venv` 內安裝 `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` 並由 `hardware.py` 註冊 DLL 路徑即可在無全域 CUDA Toolkit 環境下啟用 `CUDA + FP16`。`medium`（FP16）於 6 GB VRAM 運作流暢；`large-v3` / `large-v3-turbo` 記憶體佔用較高，若系統同時執行其他佔用 VRAM 程式可能觸發 OOM，於 `auto` 模式下將自動備援切換至 `CPU + INT8`。
- **測試結果**：
  - 自動化單元與 FFmpeg 整合測試（`.\.venv\Scripts\python.exe -m unittest discover -s tests -v`）：**41 / 41 項測試全部通過**（涵蓋 Phase 1 原有 20 項 + Phase 2 新增 21 項，含整部與多組自訂區間辨識、時間戳記偏移、所有無效時間邊界、缺少音訊、無語音片段、模型載入失敗、GPU 不可用／初始化失敗／推論期 CUDA 錯誤備援、取消辨識、JSON 讀寫往返、真實 FFmpeg 16kHz 單聲道擷取與暫存清理，以及 CLI 測試）。
  - 真實韓文音訊實測（20 秒三句韓文測試檔 `downloads/korean_test_sample.wav`，模型 `medium`）：
    - 模式 A（整部 00:00:00 ~ 00:00:20，GTX 1660 Ti `CUDA / float16`）：首次下載模型 + 推論總耗時 `42.24s`（純推論 `4.2s`），準確辨識全部 3 句韓文（`0.0~6.0`、`6.0~13.0`、`13.0~19.0`）。
    - 模式 B（自訂區間 `00:00:06 ~ 00:00:13`，GTX 1660 Ti `CUDA / float16`，讀取快取）：總耗時 `5.84s`，僅辨識該區間第 2 句，時間戳記正確偏移為 `6.0 ~ 12.0`。
    - 模式 B（自訂區間 `00:00:13 ~ 00:00:20`，`CPU / int8`）：總耗時 `9.76s`，僅辨識該區間第 3 句，時間戳記正確偏移為 `13.0 ~ 18.0`。
    - 靜音音訊實測（3 秒靜音 WAV，`CUDA / float16`）：安全回傳空片段 `segments = ()`，未發生崩潰。
- **尚未解決問題與注意事項**：
  1. 自訂時間範圍目前採精確區間切割，若使用者起訖時間剛好落在句子中間，首尾文字可能不完整；後續階段可加入可設定之前後文緩衝秒數（context padding）。
  2. 目前以 segment 級別時間戳記為主；若 Phase 3 / Phase 5 需要更細緻的單字級高亮，可於 `TranscriptionRequest` 擴充 `word_timestamps` 選項。
  3. 首次下載大型模型若未設定 `HF_TOKEN` 環境變數，Hugging Face Hub 可能顯示匿名限速提示（不影響本機下載與快取運作）。
  4. `large-v3` 在 6 GB VRAM 環境同時開啟其他 GPU 程式時可能觸發記憶體不足，建議優先使用預設 `medium` 或 `small`，必要時改用 `--device cpu`。

