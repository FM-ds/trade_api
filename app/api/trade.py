from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
# from app.database.database import get_db
# from app.models.trade import TradeFlow
# from app.schemas.trade import TradeFlowOut
from database.database import get_db
from models.trade import TradeFlow
from schemas.trade import TradeFlowOut

router = APIRouter()

@router.get("/", response_model=List[TradeFlowOut])
def filter_trade_flows(
    exporter: Optional[int] = Query(None),
    importer: Optional[int] = Query(None),
    product_code: Optional[int] = Query(None),
    year_start: Optional[int] = Query(None),
    # year_end: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(TradeFlow)
    if exporter is not None:
        query = query.filter(TradeFlow.exporter == exporter)
    if importer is not None:
        query = query.filter(TradeFlow.importer == importer)
    if product_code is not None:
        query = query.filter(TradeFlow.product_code == product_code)
    if year_start is not None:
        query = query.filter(TradeFlow.year == year_start)
    # if year_start is not None:
        # query = query.filter(TradeFlow.year >= year_start)
    # if year_start is not None and year_end is not None:
    #     query = query.filter(TradeFlow.year > year_start).filter(TradeFlow.year < year_end)


    results = query.all()
    if not results:
        raise HTTPException(status_code=404, detail="No matching trade flows found")
    return results