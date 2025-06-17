from pydantic import BaseModel

class CountryOut(BaseModel):
    code: int
    name: str

    class Config:
        orm_mode=True