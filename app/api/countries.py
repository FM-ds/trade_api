from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
# from app.database.database import get_db
# from app.models.country import Country
# from app.schemas.country import CountryOut
from database.database import get_db
from models.country import Country
from schemas.country import CountryOut

router = APIRouter()

@router.get("/", response_model=List[CountryOut])
def get_all_countries(db: Session = Depends(get_db)):
    return db.query(Country).all()

@router.get("/code/", response_model=CountryOut)
def get_country_by_code(country_code: int, db: Session = Depends(get_db)):
    country = db.query(Country).filter(Country.country_code == country_code).first()

    if not country:
        raise HTTPException(status_code=404, detail="Country not found.")
    return country