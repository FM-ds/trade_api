from pydantic import BaseModel

class TradeFlowOut(BaseModel):
    year: int
    exporter: int
    importer: int
    product_code: int
    value: float
    quantity: float

    class Config:
        from_attributes = True