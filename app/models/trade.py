from sqlalchemy import Column, Integer, Float, PrimaryKeyConstraint, ForeignKey
from sqlalchemy.orm import relationship
# from app.database.database import Base
from database.database import Base

class TradeFlow(Base):
    __tablename__ = "trade_flows"

    year = Column(Integer, index=True)
    exporter_id = Column(Integer, ForeignKey("country_codes.country_code"), index=True)
    importer_id = Column(Integer, ForeignKey("country_codes.country_code"), index=True)
    product_code = Column(Integer, ForeignKey("product_codes.code"), index=True)
    value = Column(Float, index=True)
    quantity = Column(Float, index=True)

    exporter=relationship("Country", foreign_keys = [exporter_id])
    importer=relationship("Country", foreign_keys = [importer_id])
    product_description=relationship("Product", foreign_keys = [product_code])

    __table_args__ = (
        PrimaryKeyConstraint(year, exporter_id, importer_id, product_code),
        {},
    )