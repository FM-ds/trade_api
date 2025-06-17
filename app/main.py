from fastapi import FastAPI
from app.api import trade, countries, products
from app.database.database import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(trade.router, prefix="/trade",tags=["Trade"])
app.include_router(countries.router, prefix="/countries", tags=["Countries"])
app.include_router(products.router, prefix="/products", tags=["Products"])