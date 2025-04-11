# API Server

## Setup

1. Install `uv`

2. Run `uv add -r requirements.txt`

3. Create `.env` with the following:

```txt
DB_PASSWORD=""
LTA_ACCT_KEY=""
```

## Run

1. Run `uv run api/index.py`

## Migrate

1. Run `source .venv/bin/activate` if `uv` has created the server

2. Run `flask db migrate -m "message"`

3. Run `flask db upgrade`.
