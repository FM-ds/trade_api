from sqlalchemy import Column, Integer, String
# from app.database.database import Base
from database.database import Base

class Country(Base):
    __tablename__ = "country_codes"

    country_code = Column(Integer, primary_key=True, index=True)
    country_name = Column(String(32), unique=True, index=True)
    country_iso2 = Column(String(2),index=True)