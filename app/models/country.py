from sqlalchemy import Column, Integer, String
from app.database.database import Base

class Country(Base):
    __tablename__ = "countries"

    code = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)