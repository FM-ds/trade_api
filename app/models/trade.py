from sqlalchemy import Column, Integer, Float, String, PrimaryKeyConstraint, ForeignKey
from sqlalchemy.orm import relationship
# from app.database.database import Base
from database.database import Base

class TradeFlow(Base):
    __tablename__ = "trade_flows"

    year = Column(Integer, index=True)
    # exporter = Column(Integer, ForeignKey("country_codes.country_code"), index=True)
    exporter = Column(Integer, index=True)
    importer = Column(Integer, index=True)
    # importer = Column(Integer, ForeignKey("country_codes.country_code"), index=True)
    # product_code = Column(String, ForeignKey("product_codes.code"), index=True)
    product_code = Column(String, index=True)

    value = Column(Float)
    quantity = Column(Float)

    # exporter_name=relationship("Country", foreign_keys = [exporter])
    # importer_name=relationship("Country", foreign_keys = [importer])
    # product_description=relationship("Product", foreign_keys = [product_code])

    __table_args__ = (
        PrimaryKeyConstraint(year, exporter, importer, product_code),
        {},
    )