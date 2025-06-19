from sqlalchemy import Column, Integer, String
# from app.database.database import Base
from database.database import Base

class Product(Base):
    __tablename__ = "product_codes"

    product_code = Column(String(6), primary_key=True, index=True)
    description = Column(String, unique=True, index=True)