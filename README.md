### Enhanced Video Review System (Quick Fix) — README

Цей репозиторій містить мінімалістичний веб‑сервер та UI для ручної оцінки згенерованих відео, пошуку, а також QA‑консоль для швидкого бану «невдалих» комбінацій параметрів генерації. Рішення спроєктовано для запуску на runpod.io (L40S) з JupyterLab, тому використовує фіксовані шляхи у файловій системі RunPod.

---

### Зміст
- Огляд можливостей
- Архітектура і файли (увесь проект)
- Принцип роботи (end‑to‑end)
- Запуск на RunPod (L40S + JupyterLab)
- Локальний запуск (Windows, для розробки)
- Запуск агента (генерація/аналіз) на RunPod
- Ендпоінти API та сторінки UI
- Формати даних (JSON файли)
- Відомі обмеження та зауваження безпеки
- Поширені проблеми (Troubleshooting)

---

### Огляд можливостей
- Інтерфейс для перегляду, оцінювання та фільтрації відео.
- Підрахунок статистики (згенеровано, оцінено, очікують, тощо).
- Пошук по відео з фільтрами (оцінені/неоцінені, бани, reference, параметри генерації).
- QA‑консоль для швидкого бану комбінацій параметрів за назвою відео.
- Автоматичне оновлення простих навчальних артефактів (bandit_state, knowledge) після ручних оцінок.

---

### Архітектура і файли (увесь проект)

- Корінь:
  - `simple_web_server.py` — HTTP‑сервер і UI (ручна оцінка, пошук, QA). Обробник: `QAReviewHandler`.
  - `run_agent_qa.py` — запуск QA‑CLI (`qa.cli:main`).
  - `setup_qa_no_venv.py` — встановлення залежностей із `requirements_qa.txt` у поточний Python та ініціалізація стейт‑файлів.
  - `eva_env_base.py` — базові шляхи/логування/імпорти heavy‑модулів (GPU/ML), утиліти для агента.
  - `video_wan2_2_14B_t2v.json` — приклад workflow для ComfyUI (шлях передається в CLI).

- Довідка/залежності:
  - `requirements_qa.txt` — пін інструментів аналізу/утиліт для QA.

- Веб‑клієнт:
  - `static/review_app.js` — головна сторінка перегляду/оцінювання черги відео.
  - `static/search_page.js` — сторінка пошуку.
  - `static/qa_console.js` — QA‑консоль (бан комбінацій).
  - `static/watch_page.js` — перегляд одного відео + відправка оцінки.
  - `static/ban_wrapper.js` — кнопка бану поверх `base` UI.

- Агент та компоненти:
  - `eva_p1/` — базова логіка агента та аналітики:
    - `agent_base.py` — клас `EnhancedVideoAgentV4`: робота з ComfyUI, knowledge/manual_ratings, bandit, формування черги на рев’ю.
    - `comfy_client.py` — клієнт до ComfyUI API (`/prompt`, `/history/<id>`, тощо).
    - `multi_bandit.py` — багатовимірний bandit (UCB) + міграція старих форматів стейту + автобан поганих комбінацій.
    - інші: `analysis_config.py`, `video_analyzer.py`, `workflow.py`, `openrouter_analyzer.py`, `knowledge_analyzer.py`, `prompt_generator.py`, `scenario.py`, `workflow.py`.
  - `eva_p2/` — мержений/покращений варіант агента та CLI‑патчі:
    - `merged_agent.py` — `EnhancedVideoAgentV4Merged`, додає покращений аналіз (eva_p3) і тренування.
    - `enhanced_analyzer.py`, `enhanced_logger.py`, `detection_result.py` — типи/логер/результати.
    - `cli_patch.py` — альтернативний CLI (не обов’язковий при використанні `qa/cli.py`).
  - `eva_p3/` — поглиблений аналіз/тренування/логування:
    - `video_processor.py`, `enhanced_analyzer.py` — покращені детектори/метрики.
    - `training.py` — демонстраційний тренувальний пайплайн.
    - `logger.py`, `detection_logger.py`, `detection_types.py` — логери/типи.

- QA прошарок:
  - `qa/cli.py` — основний CLI для запуску мерженого агента з патчами QA (див. нижче «Запуск агента на RunPod»).
  - `qa/agent_namespace.py` — неймспейс для імпорту мерженого агента.
  - `qa/patches.py` — набір патчів: перенаправлення логів у `auto_state/logs_improved`, посилення правил бану, guard для OpenRouter, збагачення метрик, тощо.

#### Повна структура з описом файлів

- Корінь репозиторію:
  - `README.md` — цей файл з інструкціями.
  - `requirements_qa.txt` — залежності для QA/аналізу (numpy, opencv, sklearn, тощо).
  - `setup_qa_no_venv.py` — встановлення залежностей у поточне середовище Python, створення стейт‑JSON.
  - `run_agent_qa.py` — thin‑wrapper, який викликає `qa.cli:main` (зручний запуск агента).
  - `simple_web_server.py` — веб‑сервер (порт 8189) з UI сторінками (`/`, `/search`, `/qa`, `/watch`). Обробляє API: `/api/stats`, `/api/videos`, `/api/search`, `/api/video_details`, `/api/rate`, `/api/ban_combo`, `/video/<name>`.
  - `eva_env_base.py` — базові константи шляхів `/workspace/wan22_system/...`, ініціалізація логування, імпорти heavy‑бібліотек, прапорці доступності (GPT, TF, mediapipe, scipy, тощо).
  - `video_wan2_2_14B_t2v.json` — приклад JSON‑воркфлоу для ComfyUI (передається як `--workflow`).
  - `auto_state/` — папка стану системи (на RunPod зазвичай розміщується під `/workspace/wan22_system/auto_state`):
    - `manual_ratings.json` — ручні оцінки відео (оновлюються через POST `/api/rate`).
    - `bandit_state.json` — стан бандита: `combo_stats`, `t`, `banned_combos`.
    - `knowledge.json` — історія генерацій/метрик/параметрів, `best_score`, `best_params`.
    - `review_queue.json` — черга на рев’ю (опціонально збагачує відповіді QA API).
    - `ban_history.json`, `reference_params.json` — додаткові артефакти (за потреби).
    - `logs_improved/` — логи покращеного аналізу/тренувань: `analysis.log`, `training.log`, `merged_analysis.jsonl`, `main.log`.

- `static/` — фронтенд JavaScript:
  - `review_app.js` — логіка головної сторінки: завантаження статистики, пагінація відео через `/api/videos`, відправка оцінок на `/api/rate`.
  - `search_page.js` — запит до `/api/search`, рендер таблиці з фільтрами.
  - `qa_console.js` — підвантажує список відео та викликає `POST /api/ban_combo`.
  - `watch_page.js` — показ одного відео з `/api/video_details` та форма оцінки.
  - `ban_wrapper.js` — кнопка бану поверх `base` UI.

- `eva_p1/` — базові компоненти агента:
  - `agent_base.py` — клас `EnhancedVideoAgentV4`:
    - керує шляхами/станом (`knowledge.json`, `manual_ratings.json`, `review_queue.json`),
    - інтегрує `ComfyClient` для постановки воркфлоу в ComfyUI і очікування результатів,
    - застосовує параметри до воркфлоу (див. `workflow.py`),
    - аналізує відео (простий аналіз `VideoAnalyzer`) і поповнює `knowledge.json`,
    - керує `MultiDimensionalBandit` (вибір параметрів, оновлення reward),
    - додає нові відео до `review_queue.json` (thumbnails, пріоритет, метрики).
  - `comfy_client.py` — REST‑клієнт ComfyUI (`/prompt`, `/history/<id>`, `object_info`).
  - `multi_bandit.py` — UCB‑бандит:
    - генерує/мігрує `combo_stats`,
    - autoban «поганих» комбінацій, select/update із UCB бонусом,
    - ключ комбінації: `sampler|scheduler|fps|cfg|steps|WIDTHxHEIGHT`.
  - `analysis_config.py` — параметри аналізу відео (пороги, режими, логування).
  - `video_analyzer.py` — прості метрики (blur, exposure, blockiness, flicker) і зведений `overall`.
  - `workflow.py` — валідація вузлів воркфлоу та застосування параметрів (fps/seconds/size/sampler/scheduler/steps/cfg/seed/prefix) у JSON ComfyUI.
  - `knowledge_analyzer.py` — інсайти за історією knowledge (дефекти/quality‑покращення/рекомендації експериментів).
  - `openrouter_analyzer.py` — інтерфейс до GPT через OpenRouter (опціонально; fallback якщо GPT недоступний).
  - `prompt_generator.py` + `scenario.py` — генерація розширених промптів (еротичні сценарії, якість/дефекти, негативні промпти, текстова конвертація).

- `eva_p2/` — мержені/покращені компоненти агента:
  - `merged_agent.py` — клас `EnhancedVideoAgentV4Merged`:
    - наслідує базового агента, підключає покращений `VideoProcessor`/`EnhancedLogger` (із `eva_p3`),
    - у `run_iteration_v4` обчислює blended метрику: `0.6 * base_overall + 0.4 * deep_quality`, дописує логи JSONL.
  - `enhanced_analyzer.py`, `detection_result.py`, `enhanced_logger.py` — допоміжні типи/аналізатор/логер для покращеного пайплайну.
  - `cli_patch.py` — альтернативний CLI (перевірка ComfyUI, параметри, логування). 

- `eva_p3/` — розширений аналіз/логування/тренування:
  - `video_processor.py` — покращений аналізатор (глибші детектори: anatomy/face/artifacts/temporal/частотні ознаки), повертає розширені метрики.
  - `logger.py` — логер для покращених метрик; `detection_logger.py`, `detection_types.py` — типи й логування детекцій.
  - `training.py` — демонстраційний тренер моделей/порогів метрик (опціонально).

- `qa/` — обгортка та патчі для QA‑режиму:
  - `cli.py` — парсить аргументи (`--api`, `--workflow`, `--state-dir`, `--iterations`, `--use-enhanced-analysis`, `--train-improved`, `--openrouter-key`),
    підключає патчі з `qa/patches.py`, ініціює `EnhancedVideoAgentV4Merged` і запускає пошук/генерацію.
  - `patches.py` — патчі: перенаправлення логів у `auto_state/logs_improved`, посилення бан‑правил, guard для OpenRouter, збагачення метрик.
  - `agent_namespace.py` — зручний неймспейс для доступу до мерженого агента.

Структура (спрощено):
```
simple_web_server.py   # сервер + UI
static/
  review_app.js        # головна сторінка (перегляд/оцінка черги відео)
  search_page.js       # сторінка пошуку
  qa_console.js        # QA‑консоль (бан комбінацій)
  watch_page.js        # сторінка перегляду одного відео
  ban_wrapper.js       # оверхед‑панель бану для головної сторінки
auto_state/            # JSON‑стан (на RunPod це /workspace/wan22_system/auto_state)
```

---

### Принцип роботи (end‑to‑end)
1) Зовнішній агент/генератор зберігає відео `.mp4` у каталозі `/workspace/ComfyUI/output/`.
2) Сервер на `/` рендерить UI, який викликає:
   - `GET /api/stats` — агрегована статистика;
   - `GET /api/videos?offset&limit` — список неоцінених файлів (пагінація), збагачений даними з `knowledge.json`.
3) Користувач обирає відео, оцінює його в інтерфейсі і відправляє форму:
   - `POST /api/rate` — сервер записує оцінку в `manual_ratings.json` та оновлює `bandit_state.json` і `knowledge.json` (найкращі результати, мітки в `history`).
4) Додатково:
   - `GET /search` + `GET /api/search` — пошук по назві, статусам, параметрам, авто‑метрикам;
   - `GET /qa` + `POST /api/ban_combo` — швидкий бан «невдалих» комбінацій (за `video_name`, `combo_key` або `params`).
5) Відео віддаються як `GET /video/<name>` із каталогу відео.

---

### Запуск на RunPod (L40S + JupyterLab)

Важливо: шляхи жорстко задані під RunPod. Нічого змінювати не потрібно, лише створити каталоги (якщо їх ще немає) і запустити сервер у JupyterLab Terminal.

1) Відкрийте JupyterLab → Terminal і створіть каталоги (одноразово):
```bash
mkdir -p /workspace/ComfyUI/output
mkdir -p /workspace/wan22_system/auto_state
```

2) Запустіть сервер:
```bash
python /workspace/wan22_system/simple_web_server.py
```

3) Вивід покаже порт `8189`. У RunPod відкрийте/проксіюйте порт `8189` та перейдіть за відповідним URL.

Сторінки UI:
- Головна: `/` (перегляд та оцінка черги відео)
- Пошук: `/search`
- QA‑консоль: `/qa`
- Перегляд одного відео: `/watch?name=<video.mp4>`

Порада: переконайтесь, що появились `.mp4` у `/workspace/ComfyUI/output/` — тоді на головній сторінці з’являться відео для оцінки.

---

### Локальний запуск (Windows, для розробки)
### Запуск агента (генерація/аналіз) на RunPod

Агент (пошук параметрів, запуск воркфлоу в ComfyUI, збір метрик) запускається окремо від веб‑сервера. Рекомендована послідовність у JupyterLab Terminal:

1) Підготувати середовище (один раз на pod):
```bash
python /workspace/wan22_system/setup_qa_no_venv.py --root /workspace/wan22_system
```

2) Запустити ComfyUI API (порт за замовчуванням 8188) — залежить від вашого способу розгортання ComfyUI.

3) Запустити агента з QA‑CLI (в окремому терміналі):
```bash
python /workspace/wan22_system/run_agent_qa.py
```

`run_agent_qa.py` викликає `qa.cli:main`, який приймає аргументи:
```bash
python -m qa.cli \
  --api http://127.0.0.1:8188 \
  --workflow /workspace/wan22_system/video_wan2_2_14B_t2v.json \
  --state-dir /workspace/wan22_system/auto_state \
  --seconds 5.0 \
  --iterations 10 \
  --openrouter-key $OPENROUTER_API_KEY \
  --use-enhanced-analysis \
  --train-improved
```

Примітки:
- `--workflow` має вказувати на валідний JSON воркфлоу ComfyUI.
- `--iterations` визначає кількість ітерацій пошуку/генерації.
- Якщо ви не використовуєте GPT/OpenRouter — опустіть `--openrouter-key`.
- Покращений аналіз (`--use-enhanced-analysis`) та тренування (`--train-improved`) опціональні; за необхідності вимикайте для економії GPU/часу.

4) Паралельно тримайте запущеним веб‑сервер для ручної оцінки (`simple_web_server.py`), щоб оцінювати результати генерації:
```bash
python /workspace/wan22_system/simple_web_server.py
```

Взаємодія:
- Агент зберігає результати/стан у `/workspace/wan22_system/auto_state/` і відео у `/workspace/ComfyUI/output/`.
- Веб‑UI підхоплює нові відео, показує метрики/параметри з `knowledge.json` і приймає ручні оцінки, що впливають на подальший пошук (bandit).

---

### Повна покрокова інсталяція

#### RunPod (L40S + JupyterLab)

1) Запустіть pod із JupyterLab та GPU L40S (образ з Python + PyTorch 2.2).
2) Відкрийте JupyterLab → Terminal.
3) Розмістіть репозиторій у `/workspace/wan22_system` (або скопіюйте туди вміст):
```bash
cd /workspace
mkdir -p wan22_system
# Якщо потрібно, клонувати:
# git clone <repo_url> wan22_system
```
4) Створіть каталоги для відео/стану (на випадок, якщо їх ще немає):
```bash
mkdir -p /workspace/ComfyUI/output
mkdir -p /workspace/wan22_system/auto_state
```
5) Встановіть залежності та ініціалізуйте стейт‑файли:
```bash
python /workspace/wan22_system/setup_qa_no_venv.py --root /workspace/wan22_system
```
6) Запустіть ComfyUI API (порт 8188) — згідно з інструкцією вашого розгортання ComfyUI.
7) Запустіть веб‑сервер (порт 8189) в окремому терміналі:
```bash
python /workspace/wan22_system/simple_web_server.py
```
8) Відкрийте у RunPod URL для порту 8189 (головна сторінка з UI).
9) Запустіть агента (ще один термінал):
```bash
python /workspace/wan22_system/run_agent_qa.py
# або з явними параметрами:
python -m qa.cli \
  --api http://127.0.0.1:8188 \
  --workflow /workspace/wan22_system/video_wan2_2_14B_t2v.json \
  --state-dir /workspace/wan22_system/auto_state \
  --seconds 5.0 \
  --iterations 10 \
  --use-enhanced-analysis
```
10) У процесі генерації нові `.mp4` з’являтимуться у `/workspace/ComfyUI/output/`, і ви зможете їх оцінювати через UI.

Опційно: для GPT‑аналізу задайте `--openrouter-key $OPENROUTER_API_KEY` та додайте бібліотеку OpenAI (вже є у `requirements_qa.txt`).

#### Windows (для локальної розробки)

1) Встановіть Python 3.10+ (рекомендовано 3.10) і `pip`.
2) Створіть аналоги RunPod‑шляхів:
```powershell
New-Item -ItemType Directory -Force -Path C:\workspace\ComfyUI\output | Out-Null
New-Item -ItemType Directory -Force -Path C:\workspace\wan22_system\auto_state | Out-Null
```
3) Встановіть залежності:
```powershell
cd C:\IT\GItRepo\try294
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_qa.txt
```
4) Запустіть локальний ComfyUI API на `http://127.0.0.1:8188` (якщо доступно).
5) Запустіть веб‑сервер:
```powershell
python simple_web_server.py
```
6) (Опц.) Запустіть агента:
```powershell
python run_agent_qa.py
# або:
python -m qa.cli --api http://127.0.0.1:8188 --workflow C:\IT\GItRepo\try294\video_wan2_2_14B_t2v.json --state-dir C:\workspace\wan22_system\auto_state --iterations 5
```
7) Покладіть `.mp4` у `C:\workspace\ComfyUI\output` — вони з’являться в UI на `http://127.0.0.1:8189/`.


Шляхи у коді вказані під RunPod. Для локальної перевірки найпростіше відтворити ці шляхи у Windows (створити аналоги каталогів на диску `C:`):

```powershell
New-Item -ItemType Directory -Force -Path C:\workspace\ComfyUI\output | Out-Null
New-Item -ItemType Directory -Force -Path C:\workspace\wan22_system\auto_state | Out-Null

cd C:\IT\GItRepo\try294
python simple_web_server.py
```

Далі відкрийте `http://127.0.0.1:8189/` у браузері. Покладіть `.mp4` у `C:\workspace\ComfyUI\output` — вони з’являться в UI.

Примітка: для повної кросплатформеності можна параметризувати шляхи через ENV/CLI, але поточна версія свідомо спрощена під RunPod.

---

### Ендпоінти API та сторінки UI

Сторінки UI:
- `GET /` — головна сторінка (JS: `static/review_app.js`)
- `GET /search` — сторінка пошуку (JS: `static/search_page.js`)
- `GET /qa` — QA‑консоль (JS: `static/qa_console.js`)
- `GET /watch?name=<video.mp4>` — перегляд одного відео (JS: `static/watch_page.js`)

API:
- `GET /api/stats` → агрегована статистика.
- `GET /api/videos?offset=<int>&limit=<int>` → список неоцінених відео (пагінація), збагачений даними з `knowledge.json`.
- `GET /video/<name>` → сам файл відео.
- `POST /api/rate` → зберегти ручну оцінку.
- `GET /api/search` → пошук по назві/параметрах/статусах.
- `GET /api/video_details?name=<video.mp4>` → деталі з knowledge/manual + факт наявності файлу.
- `POST /api/ban_combo` (лише в `QAReviewHandler`) → бан зазначеної комбо.

Приклади запитів:

`POST /api/rate` (JSON):
```json
{
  "video_name": "sample_1699999999.mp4",
  "rating": {
    "overall_quality": 7,
    "visual_quality": 8,
    "motion_quality": 6,
    "prompt_adherence": 7,
    "creativity": 7,
    "technical_quality": 7,
    "is_reference": true,
    "comments": "Достатньо добре"
  }
}
```

`POST /api/ban_combo` (JSON — один із варіантів):
```json
{ "video_name": "sample_1699999999.mp4" }
```
або
```json
{ "combo_key": "sampler|scheduler|fps|cfg|steps|WIDTHxHEIGHT" }
```
або
```json
{
  "params": {"sampler": "euler", "scheduler": "karras", "fps": 20, "cfg_scale": 7, "steps": 25, "width": 768, "height": 432}
}
```

---

### Формати даних (JSON файли)

`manual_ratings.json` (ключ — ім’я файлу відео):
```json
{
  "sample_1699999999.mp4": {
    "rating": {
      "overall_quality": 7,
      "visual_quality": 8,
      "motion_quality": 6,
      "prompt_adherence": 7,
      "creativity": 7,
      "technical_quality": 7,
      "is_reference": true,
      "comments": "Достатньо добре"
    },
    "timestamp": 1700000000.0,
    "rated_at": "2025-09-10 12:00:00"
  }
}
```

`knowledge.json` (спрощено):
```json
{
  "best_score": 8.5,
  "best_params": { "prompt": "...", "params": {"fps": 20, "width": 768, "height": 432}, "combo": ["sampler","scheduler"], "manual_rating": {"overall_quality": 9} },
  "history": [
    {
      "video": "/workspace/ComfyUI/output/sample_1699999999.mp4",
      "timestamp": 1699999999,
      "prompt": "...",
      "params": {"fps": 20, "width": 768, "height": 432, "sampler": "euler", "scheduler": "karras"},
      "combo": ["euler", "karras"],
      "metrics": {"overall": 0.72}
    }
  ]
}
```

`bandit_state.json` (мінімум для лічильників):
```json
{ "arms": [], "N": [], "S": [], "t": 42, "banned_combos": ["euler|karras|20|7|25|768x432"] }
```

`review_queue.json` (необов’язковий, збагачує відповіді в QA‑режимі):
```json
{ "pending": [ { "original_path": "/workspace/ComfyUI/output/sample_1699999999.mp4", "auto_metrics": {"overall": 0.7}, "params": {"fps": 20}, "combo": ["euler","karras"] } ] }
```

---

### Відомі обмеження та зауваження безпеки
- Шляхи до каталогів жорстко зафіксовані під RunPod (див. код у `__init__`). Для продуктивних сценаріїв рекомендується винести у змінні середовища/CLI.
- Порт сервера фіксований: `8189` (див. низ файлу `simple_web_server.py`).
- Аутентифікація/авторизація відсутні. Використовуйте у довіреному середовищі (RunPod pod/Jupyter), не публікуйте напряму у відкритий інтернет.

---

### Поширені проблеми (Troubleshooting)
- Немає відео на головній сторінці:
  - Переконайтесь, що `.mp4` лежать у `/workspace/ComfyUI/output/` (або у відповідному локальному каталозі).
- `/api/stats` показує нулі:
  - Це нормально для «чистого» старту; після генерацій/оцінок значення зміняться.
- Помилки доступу/шляхів:
  - Створіть каталоги згідно інструкції. RunPod: `/workspace/ComfyUI/output` і `/workspace/wan22_system/auto_state`.
- 404 для `/video/<name>`:
  - Перевірте, що файл існує під точним ім’ям і у правильному каталозі.
- CORS/браузерні помилки:
  - Сервер додає `Access-Control-Allow-Origin: *` для API; якщо ви змінювали код/хостинг, перевірте заголовки.

---

Готово! Якщо потрібно, можу доповнити README інструкціями з інтеграції конкретного агента/потоку генерації або додати параметризацію шляхів для кросплатформеності.


