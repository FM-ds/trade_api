from pydantic import BaseModel

class TradeFlowOut(BaseModel):
    id: int
    exporter: int
    importer: int
    product_code: int
    value: float

    class Config:
        orm_mode = True