from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import duckdb
import pandas as pd
import time

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from datetime import datetime
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {
        "request": request,
        "form_data": {
            "query_type": "1",
            "exporter": 156,
            "importer": 826,
            "product": 950300,
            "year_from": 2017,
            "year_to": 2022,
        },
        "table": None,
        "elapsed": None
    })

@app.post("/query", response_class=HTMLResponse)
async def run_query(
    request: Request,
    query_type: str = Form(...),
    exporter: int = Form(...),
    importer: int = Form(None),
    product: int = Form(...),
    year_from: int = Form(...),
    year_to: int = Form(...)
):
    start_time = time.time()

    if query_type == "1":
        query = f"""
            SELECT * FROM 'baci_hs17_2017_2022.parquet'
            WHERE exporter = {exporter}
              AND importer = {importer}
              AND product = {product}
              AND year BETWEEN {year_from} AND {year_to}
            LIMIT 100
        """
    elif query_type == "2":
        query = f"""
            SELECT year, importer, SUM(value) AS total_value
            FROM 'baci_hs17_2017_2022.parquet'
            WHERE exporter = {exporter}
              AND product = {product}
              AND year BETWEEN {year_from} AND {year_to}
            GROUP BY year, importer
            ORDER BY year, total_value DESC
            LIMIT 100
        """
    elif query_type == "3":
        query = f"""
            SELECT product, SUM(value) AS total_value
            FROM 'baci_hs17_2017_2022.parquet'
            WHERE exporter = {exporter}
            GROUP BY product
            ORDER BY total_value DESC
            LIMIT 20
        """
    elif query_type == "4":
        query = f"""
            SELECT importer, SUM(value) AS total_value
            FROM 'baci_hs17_2017_2022.parquet'
            WHERE exporter = {exporter}
              AND year BETWEEN {year_from} AND {year_to}
            GROUP BY importer
            ORDER BY total_value DESC
            LIMIT 50
        """
    elif query_type == "5":
        query = f"""
            WITH yearly_totals AS (
                SELECT importer, product, year, SUM(value) AS total_value
                FROM 'baci_hs17_2017_2022.parquet'
                WHERE exporter = {exporter}
                  AND year IN (2017, 2022)
                GROUP BY importer, product, year
            ),
            pivoted AS (
                SELECT 
                    importer,
                    product,
                    MAX(CASE WHEN year = 2017 THEN total_value ELSE NULL END) AS value_2017,
                    MAX(CASE WHEN year = 2022 THEN total_value ELSE NULL END) AS value_2022
                FROM yearly_totals
                GROUP BY importer, product
            ),
            differences AS (
                SELECT importer, product, value_2017, value_2022,
                       ROUND(100.0 * (value_2022 - value_2017) / value_2017, 2) AS pct_growth
                FROM pivoted
                WHERE value_2017 IS NOT NULL AND value_2022 IS NOT NULL AND value_2017 >= 1000
            )
            SELECT *
            FROM differences
            ORDER BY pct_growth DESC
            LIMIT 10
        """
    else:
        query = "SELECT 1"

    df = duckdb.query(query).to_df()
    elapsed = round(time.time() - start_time, 3)
    table_html = df.to_html(index=False)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "table": table_html,
        "elapsed": elapsed,
        "form_data": {
            "query_type": query_type,
            "exporter": exporter,
            "importer": importer if importer is not None else '',
            "product": product,
            "year_from": year_from,
            "year_to": year_to,
        }
    })

with open("products.json", "r") as f:
    PRODUCTS = json.load(f)
with open("countries.json", "r") as f:
    COUNTRIES = json.load(f)

@app.get("/api/products")
async def search_products(
    search: Optional[str] = None,
    limit: int = Query(50, le=100)
) -> List[dict]:
    """Returns products matching search term."""
    
    if not search:
        return PRODUCTS[:limit]
    
    search_term = str(search).lower()
    results = []
    
    for product in PRODUCTS:
        if (search_term in str(product["code"]) or 
            search_term in product["description"].lower()):
            results.append(product)
            
        if len(results) >= limit:
            break
    
    return results

@app.get("/api/countries")
async def search_products(
    search: Optional[str] = None,
    limit: int = Query(50, le=10000)
) -> List[dict]:
    """Returns products matching search term."""
    
    if not search:
        return COUNTRIES[:limit]
    
    search_term = str(search).lower()
    results = []
    
    for product in COUNTRIES:
        if (search_term in str(product["code"]) or 
            search_term in product["country_name"].lower()):
            results.append(product)
            
        if len(results) >= limit:
            break
    
    return results



class TradeRecord(BaseModel):
    """Individual trade data record."""
    product_code: str      # HS/CN8 product code
    product: str      # Product description
    year: int             # Trade year
    partner: str          # Trading partner country
    trade_flow: str       # "imports" or "exports"
    value: float          # Trade value in USD
    quantity: float       # Quantity traded
    unit: str            # Unit of measurement (typically "kg")


class TradeDataResponse(BaseModel):
    """Structured trade data response."""
    total_records: int          # Total number of matching records
    page: int                   # Current page number
    page_size: int             # Records per page
    total_pages: int           # Total pages available
    data: List[TradeRecord]    # Array of trade records for current page
    execution_time_ms: float   # Query execution time in milliseconds



@app.get("/api/trade-query", response_model=TradeDataResponse)
async def query_trade_data(
    trade_type: str = Query("imports", regex="^(imports|exports|all)$"),
    product_codes: str = Query("950300", description="Comma-separated product codes"),
    from_country: str = Query("156", description="Origin country"),
    to_country: str = Query("everywhere", description="Destination country"),
    year_from: int = Query(2020, ge=2000, le=2024),
    year_to: int = Query(2022, ge=2000, le=2024),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=10, le=1000)
):
    """Returns trade data based on query parameters."""
    
    start_time = datetime.now()
    
    if year_to < year_from:
        raise HTTPException(status_code=400, detail="year_to must be >= year_from")
    
    product_list = [code.strip() for code in product_codes.split(",") if code.strip()]
    if not product_list:
        raise HTTPException(status_code=400, detail="At least one product code required")
    
    try:
        aggregate_from = from_country == "world"
        trade_flow_value = trade_type if trade_type != "all" else "imports"
        
        # Build parameters list in order
        params = []
        
        if aggregate_from:
            # Aggregated query
            select_clause = """
            SELECT 
                product,
                product,
                year,
                'World' as partner,
                ? as trade_flow,
                SUM(value) as value,
                SUM(quantity) as quantity,
                'kg' as unit
            """
            params.append(trade_flow_value)
            group_clause = "GROUP BY product, product, year"
        else:
            # Individual records
            partner_field = "importer" if trade_type == "exports" else "exporter"
            select_clause = f"""
            SELECT 
                product,
                product,
                year,
                {partner_field} as partner,
                ? as trade_flow,
                value as value,
                quantity as quantity,
                'kg' as unit
            """
            params.append(trade_flow_value)
            group_clause = ""
        
        # Build WHERE conditions
        where_conditions = []
        
        # Year filter
        where_conditions.append("year BETWEEN ? AND ?")
        params.extend([year_from, year_to])
        
        # Product codes
        product_placeholders = ",".join(["?" for _ in product_list])
        where_conditions.append(f"product IN ({product_placeholders})")
        params.extend(product_list)
        
        # Country filters
        if from_country not in ["everywhere", "world"]:
            where_conditions.append("exporter = ?")
            params.append(from_country)
        
        if to_country not in ["everywhere", "world"]:
            where_conditions.append("importer = ?")
            params.append(to_country)
        
        # Complete queries
        base_from = "FROM 'baci_hs17_2017_2022.parquet'"
        where_clause = f"WHERE {' AND '.join(where_conditions)}"
        
        # Count query
        if group_clause:
            count_query = f"""
            SELECT COUNT(*) as total FROM (
                SELECT 1 {base_from} {where_clause} {group_clause}
            )
            """
            count_params = list(params)
        else:
            count_query = f"SELECT COUNT(*) as total {base_from} {where_clause}"
            # Remove the trade_flow param for count if not grouped
            count_params = list(params)
            # Remove the first param (trade_flow) for count query if not grouped
            # because it's only used in SELECT, not WHERE
            if count_params:
                count_params = count_params[1:]
        
        # Data query
        offset = (page - 1) * page_size
        data_query = f"""
        {select_clause}
        {base_from}
        {where_clause}
        {group_clause}
        ORDER BY year DESC, value DESC
        LIMIT ? OFFSET ?
        """
        
        conn = duckdb.connect()
        
        # Use a separate params list for count and data queries
        data_params = list(params) + [page_size, offset]  # Add LIMIT and OFFSET for data query
        
        total_result = conn.execute(count_query, count_params).fetchone()
        total_records = total_result[0] if total_result else 0
        
        data_result = conn.execute(data_query, data_params).fetchall()
        
        conn.close()
        
        # Convert results
        data = [
            TradeRecord(
                product_code=str(row[0]),
                product=str(row[1]),
                year=row[2],
                partner=str(row[3]),
                trade_flow=row[4],
                value=float(row[5]) if row[5] else 0.0,
                quantity=float(row[6]) if row[6] else 0.0,
                unit=row[7]
            )
            for row in data_result
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        total_pages = (total_records + page_size - 1) // page_size
        
        return TradeDataResponse(
            total_records=total_records,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            data=data,
            execution_time_ms=round(execution_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query failed: {str(e)}")