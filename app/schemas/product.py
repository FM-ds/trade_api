from pydantic import BaseModel

class ProductOut(BaseModel):
    code: str
    description: str

    class Config:
        from_attributes=True