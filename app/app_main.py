from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

# from app.api import trade, countries, products
# from app.database.database import Base, engine
from api import trade, countries, products
from database.database import Base, engine


Base.metadata.create_all(bind=engine)

app = FastAPI(title="Trade Data API")
app.include_router(trade.router, prefix="/trade",tags=["Trade"])
app.include_router(countries.router, prefix="/countries", tags=["Countries"])
app.include_router(products.router, prefix="/products", tags=["Products"])

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")