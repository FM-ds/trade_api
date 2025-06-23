from sqlalchemy import Column, Integer, Float, PrimaryKeyConstraint
# from app.database.database import Base
from database.database import Base

class TradeFlow(Base):
    __tablename__ = "trade_flows"

    year = Column(Integer, index=True)
    exporter = Column(Integer, index=True)
    importer = Column(Integer, index=True)
    product_code = Column(Integer, index=True)
    value = Column(Float, index=True)
    quantity = Column(Float, index=True)

    __table_args__ = (
        PrimaryKeyConstraint(year, exporter, importer, product_code),
        {},
    )