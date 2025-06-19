from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
# from app.database.database import get_db
# from app.models.product import Product
# from app.schemas.product import ProductOut
from database.database import get_db
from models.product import Product
from schemas.product import ProductOut

router = APIRouter()

@router.get("/products/", response_model=List[ProductOut])
def get_all_products(db: Session = Depends(get_db)):
    return db.query(Product).all()

@router.get("/products/{code}", response_model=ProductOut)
def get_product_by_code(product_code: int, db: Session = Depends(get_db)):
    country = db.query(Product).filter(Product.code == product_code).first()

    if not country:
        raise HTTPException(status_code=404, detail="Product not found.")
    return country