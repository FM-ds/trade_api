from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import json

with open("creds.json", "r") as f:
    creds = json.load(f)
    pc = Pinecone(
        api_key=creds["pinecone"]
    )
    DB_URL = creds["db"]

# engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
engine = create_engine(DB_URL)

product_codes_index = pc.Index("product-codes-index")
country_index = pc.Index("country-codes-index")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()