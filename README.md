# fam_photo_v1
React + FastAPI + HF

## backend env
- `cd backend`
- env
- `uv sync`
- run
- `uv run uvicorn main:app --host "0.0.0.0" --port "8000"`
- watch/dev
- `uv run uvicorn main:app --host "0.0.0.0" --port "8000" --reload`
- multiple workers
- `uv run uvicorn main:app --host "0.0.0.0" --port "8000" --workers 4`

## frontend
- `cd app`
- env
- `npm install`
- run
- `npm start`
