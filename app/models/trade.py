from sqlalchemy import Column, Integer, Float
from app.database.database import Base

class TradeFlow(Base):
    __tablename__ = "trade_flows"

    id = Column(Integer, primary_key=True, index=True)
    exporter = Column(Integer, index=True)
    importer = Column(Integer, index=True)
    product_code = Column(Integer, index=True)
    value = Column(Float)