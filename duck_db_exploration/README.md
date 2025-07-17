You'll need baci_hs17_2017_2022.parquet in this directory to get it to work. 

Build a venv from the requirements.txt file:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then launch main.py as always, e.g. with uvicorn:

```bash
uvicorn main:app --reload
```