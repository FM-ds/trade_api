from pydantic import BaseModel

class CountryOut(BaseModel):
    country_code: int
    country_name: str

    class Config:
        from_attributes=True