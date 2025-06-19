from sqlalchemy import Column, Integer, Float
# from app.database.database import Base
from database.database import Base

class TradeFlow(Base):
    __tablename__ = "trade_flows"

    product_code = Column(Integer, primary_key=True, index=True)
    year = Column(Integer, index=True)
    exporter = Column(Integer, index=True)
    importer = Column(Integer, index=True)
    value = Column(Float, index=True)
    quantity = Column(Float, index=True)