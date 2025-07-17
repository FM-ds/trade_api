from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import duckdb
import pandas as pd
import time
import httpx
import asyncio
import logging

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime
import json
from fastapi.middleware.cors import CORSMiddleware

class TariffCache:
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_duration = 24 * 60 * 60  # 24 hours in seconds
    
    def _get_cache_key(self, endpoint: str, params: dict = None) -> str:
        """Generate a unique cache key for the endpoint and parameters."""
        key = endpoint
        if params:
            # Sort params for consistent key generation
            sorted_params = sorted(params.items())
            key += "?" + "&".join([f"{k}={v}" for k, v in sorted_params])
        return key
    
    def get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Get cached data if it exists and is still valid."""
        cache_key = self._get_cache_key(endpoint, params)
        
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            cached_time = cache_entry.get("timestamp", 0)
            current_time = time.time()
            
            # Check if cache is still valid (within 24 hours)
            if current_time - cached_time < self._cache_duration:
                logger.info(f"Cache hit for {cache_key}")
                return cache_entry.get("data")
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
                logger.info(f"Cache expired for {cache_key}")
        
        return None
    
    def set(self, endpoint: str, data: dict, params: dict = None):
        """Store data in cache with current timestamp."""
        cache_key = self._get_cache_key(endpoint, params)
        self._cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }
        logger.info(f"Cached data for {cache_key}")
    
    def clear(self):
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        current_time = time.time()
        valid_entries = 0
        expired_entries = 0
        
        for cache_entry in self._cache.values():
            cached_time = cache_entry.get("timestamp", 0)
            if current_time - cached_time < self._cache_duration:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_duration_hours": self._cache_duration / 3600
        }

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# GOV.UK Trade Tariff API configuration
TRADE_TARIFF_BASE_URL = "https://www.trade-tariff.service.gov.uk/api/v2"
USER_AGENT = "HMT-Hackathon-Trade-API / 1.0 (Trade data analysis tool)"

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Add new Pydantic models for Trade Tariff API
class TariffSection(BaseModel):
    """Trade Tariff section data."""
    id: str
    title: str
    position: int
    section_note: Optional[str] = None

class TariffChapter(BaseModel):
    """Trade Tariff chapter data."""
    goods_nomenclature_item_id: str
    description: str
    formatted_description: str
    chapter_note: Optional[str] = None

class TariffCommodity(BaseModel):
    """Trade Tariff commodity data."""
    goods_nomenclature_item_id: str
    description: str
    formatted_description: str
    number_indents: int = 0
    producline_suffix: str = ""
    leaf: bool = True
    goods_nomenclature_class: str = "commodity"

class TariffMeasure(BaseModel):
    """Trade Tariff measure data."""
    id: str
    measure_type: str
    duty_expression: str
    geographical_area: str
    effective_start_date: str
    effective_end_date: Optional[str] = None

class TariffDataResponse(BaseModel):
    """Combined tariff data response."""
    commodity: TariffCommodity
    measures: List[TariffMeasure]
    execution_time_ms: float

# Create global cache instance
tariff_cache = TariffCache()

# Helper function to make API calls to Trade Tariff API
async def fetch_trade_tariff_data(endpoint: str, params: dict = None) -> dict:
    """Fetch data from GOV.UK Trade Tariff API with 24-hour caching."""
    
    # Try to get from cache first
    cached_data = tariff_cache.get(endpoint, params)
    if cached_data:
        return cached_data
    
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            url = f"{TRADE_TARIFF_BASE_URL}/{endpoint}"
            logger.info(f"Requesting: {url} with params: {params}")
            
            response = await client.get(
                url,
                headers=headers,
                params=params or {},
                timeout=30.0
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            response.raise_for_status()
            
            # Check if response is actually JSON
            content_type = response.headers.get('content-type', '')
            if 'application/json' not in content_type:
                raise HTTPException(
                    status_code=502, 
                    detail=f"Trade Tariff API returned non-JSON response. Content-Type: {content_type}"
                )
            
            data = response.json()
            
            # Cache the successful response
            tariff_cache.set(endpoint, data, params)
            
            return data
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise HTTPException(
                status_code=502 if response.status_code != 404 else 404, 
                detail=f"Trade Tariff API error: {str(e)}"
            )
        except ValueError as e:
            logger.error(f"JSON parsing error: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Trade Tariff API returned invalid JSON: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error when calling Trade Tariff API: {str(e)}"
            )

# Add cache management endpoints
@app.get("/api/tariff/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    return tariff_cache.get_cache_stats()

@app.post("/api/tariff/cache/clear")
async def clear_cache():
    """Clear all cached tariff data."""
    tariff_cache.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/api/tariff/cache/health")
async def cache_health():
    """Check cache health and cleanup expired entries."""
    stats = tariff_cache.get_cache_stats()
    
    # Clean up expired entries
    current_time = time.time()
    expired_keys = []
    
    for key, cache_entry in tariff_cache._cache.items():
        cached_time = cache_entry.get("timestamp", 0)
        if current_time - cached_time >= tariff_cache._cache_duration:
            expired_keys.append(key)
    
    # Remove expired entries
    for key in expired_keys:
        del tariff_cache._cache[key]
    
    return {
        "cache_stats": stats,
        "cleaned_expired_entries": len(expired_keys),
        "remaining_entries": len(tariff_cache._cache)
    }

# New endpoints for Trade Tariff API
@app.get("/api/tariff/sections")
async def get_tariff_sections() -> List[TariffSection]:
    """Get all trade tariff sections (cached for 24 hours)."""
    start_time = datetime.now()
    
    data = await fetch_trade_tariff_data("sections")
    
    sections = []
    for section_data in data.get("data", []):
        sections.append(TariffSection(
            id=section_data["id"],
            title=section_data["attributes"]["title"],
            position=section_data["attributes"]["position"],
            section_note=section_data["attributes"].get("section_note")
        ))
    
    return sections

@app.get("/api/tariff/chapters")
async def get_tariff_chapters(section_id: Optional[str] = Query(None)) -> List[TariffChapter]:
    """Get trade tariff chapters, optionally filtered by section."""
    
    params = {}
    if section_id:
        params["filter[section_id]"] = section_id
    
    data = await fetch_trade_tariff_data("chapters", params)
    
    chapters = []
    for chapter_data in data.get("data", []):
        attrs = chapter_data.get("attributes", {})
        
        chapters.append(TariffChapter(
            id=chapter_data["id"],
            goods_nomenclature_item_id=attrs.get("goods_nomenclature_item_id", ""),
            # Use formatted_description first, then description, then fallback
            description=attrs.get("formatted_description", 
                                attrs.get("description", 
                                        attrs.get("short_description", "No description available"))),
            chapter_note=attrs.get("chapter_note"),
            section_id=attrs.get("section_id")
        ))
    
    return chapters

@app.get("/api/tariff/commodities/{commodity_code}")
async def get_commodity_details(
    commodity_code: str,
    country: Optional[str] = Query(None, description="Country code for specific measures"),
    as_of: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    bypass_cache: bool = Query(False, description="Bypass cache and fetch fresh data")
) -> TariffDataResponse:
    """Get detailed commodity information including measures (cached for 24 hours)."""
    start_time = datetime.now()
    
    # Validate commodity code format
    if not commodity_code.isdigit() or len(commodity_code) < 6:
        raise HTTPException(
            status_code=400,
            detail="Commodity code must be at least 6 digits"
        )
    
    # Pad to 10 digits if needed
    commodity_code = commodity_code.ljust(10, '0')
    
    params = {}
    if country:
        params["filter[geographical_area_id]"] = country
    if as_of:
        params["as_of"] = as_of
    
    # If bypass_cache is True, clear this specific cache entry
    if bypass_cache:
        cache_key = tariff_cache._get_cache_key(f"commodities/{commodity_code}", params)
        if cache_key in tariff_cache._cache:
            del tariff_cache._cache[cache_key]
            logger.info(f"Bypassed cache for {cache_key}")
    
    try:
        data = await fetch_trade_tariff_data(f"commodities/{commodity_code}", params)
        
        # Check if we got valid data structure
        if "data" not in data:
            raise HTTPException(
                status_code=404,
                detail=f"Commodity {commodity_code} not found"
            )
        
        # Extract commodity data - handle both single object and array responses
        commodity_data = data["data"]
        if isinstance(commodity_data, list):
            if not commodity_data:
                raise HTTPException(status_code=404, detail=f"Commodity {commodity_code} not found")
            commodity_data = commodity_data[0]
        
        if "attributes" not in commodity_data:
            raise HTTPException(
                status_code=404,
                detail=f"Invalid commodity data structure for {commodity_code}"
            )
        
        attrs = commodity_data["attributes"]
        commodity = TariffCommodity(
            goods_nomenclature_item_id=attrs.get("goods_nomenclature_item_id", commodity_code),
            description=attrs.get("description", "Unknown"),
            formatted_description=attrs.get("formatted_description", attrs.get("description", "Unknown")),
            number_indents=attrs.get("number_indents", 0),
            producline_suffix=attrs.get("producline_suffix", "80"),
            leaf=attrs.get("leaf", True),
            goods_nomenclature_class=attrs.get("goods_nomenclature_class", "commodity")
        )
        
        # Extract measures from included data
        measures = []
        included_data = data.get("included", [])
        
        for item in included_data:
            if item.get("type") == "measure":
                try:
                    measure_attrs = item.get("attributes", {})
                    
                    # Extract measure type
                    measure_type = "Unknown"
                    if "relationships" in item:
                        measure_type_data = item["relationships"].get("measure_type", {}).get("data", {})
                        if measure_type_data:
                            # Find the measure type in included data
                            measure_type_id = measure_type_data.get("id")
                            for inc_item in included_data:
                                if inc_item.get("type") == "measure_type" and inc_item.get("id") == measure_type_id:
                                    measure_type = inc_item.get("attributes", {}).get("description", "Unknown")
                                    break
                    
                    # Extract duty expression
                    duty_expression = "N/A"
                    if "relationships" in item:
                        duty_expr_data = item["relationships"].get("duty_expression", {}).get("data", {})
                        if duty_expr_data:
                            duty_expr_id = duty_expr_data.get("id")
                            for inc_item in included_data:
                                if inc_item.get("type") == "duty_expression" and inc_item.get("id") == duty_expr_id:
                                    duty_expression = inc_item.get("attributes", {}).get("base", "N/A")
                                    break
                    
                    # Extract geographical area
                    geographical_area = "Unknown"
                    if "relationships" in item:
                        geo_data = item["relationships"].get("geographical_area", {}).get("data", {})
                        if geo_data:
                            geo_id = geo_data.get("id")
                            for inc_item in included_data:
                                if inc_item.get("type") == "geographical_area" and inc_item.get("id") == geo_id:
                                    geographical_area = inc_item.get("attributes", {}).get("description", "Unknown")
                                    break
                    
                    measures.append(TariffMeasure(
                        id=item.get("id", "unknown"),
                        measure_type=measure_type,
                        duty_expression=duty_expression,
                        geographical_area=geographical_area,
                        effective_start_date=measure_attrs.get("effective_start_date", ""),
                        effective_end_date=measure_attrs.get("effective_end_date")
                    ))
                except Exception as e:
                    logger.warning(f"Error processing measure {item.get('id', 'unknown')}: {e}")
                    continue
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return TariffDataResponse(
            commodity=commodity,
            measures=measures,
            execution_time_ms=round(execution_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing commodity {commodity_code}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing commodity data: {str(e)}"
        )

@app.get("/api/tariff/search")
async def search_commodities(
    query: str = Query(..., description="Search term for commodities"),
    exact_match: bool = Query(False, description="Whether to search for exact matches only")
) -> List[TariffCommodity]:
    """Search for commodities by description or code."""
    
    if len(query.strip()) < 2:
        raise HTTPException(
            status_code=400,
            detail="Search query must be at least 2 characters long"
        )
    
    params = {
        "q": query,
        "exact_match": str(exact_match).lower()
    }
    
    try:
        # Use search_references endpoint instead
        data = await fetch_trade_tariff_data("search_references", params)
        
        commodities = []
        search_data = data.get("data", [])
        
        for item in search_data:
            if item.get("type") == "search_reference":
                attrs = item.get("attributes", {})
                
                # Get the referenced commodity code
                referenced_id = attrs.get("referenced_id", "")
                referenced_class = attrs.get("referenced_class", "")
                
                # Convert heading/chapter references to 10-digit codes
                if referenced_class == "Heading" and len(referenced_id) == 4:
                    referenced_id = referenced_id + "000000"
                elif referenced_class == "Chapter" and len(referenced_id) == 2:
                    referenced_id = referenced_id + "00000000"
                elif len(referenced_id) < 10:
                    referenced_id = referenced_id.ljust(10, '0')
                
                commodities.append(TariffCommodity(
                    goods_nomenclature_item_id=referenced_id,
                    description=attrs.get("title", ""),
                    formatted_description=attrs.get("title", ""),
                    number_indents=0,
                    producline_suffix=attrs.get("productline_suffix", "80"),
                    leaf=True,
                    goods_nomenclature_class=referenced_class.lower() if referenced_class else "commodity"
                ))
        
        return commodities
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching commodities with query '{query}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching commodities: {str(e)}"
        )

# Add a test endpoint to check API connectivity
@app.get("/api/tariff/test")
async def test_tariff_api():
    """Test endpoint to check Trade Tariff API connectivity."""
    try:
        # Try to fetch sections as a simple test
        data = await fetch_trade_tariff_data("sections")
        return {
            "status": "success",
            "message": "Trade Tariff API is accessible",
            "sections_count": len(data.get("data", []))
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Enhanced trade query endpoint that includes tariff data
@app.get("/api/enhanced-trade-query", response_model=dict)
async def enhanced_trade_query(
    trade_type: str = Query("imports", regex="^(imports|exports|all)$"),
    product_codes: str = Query("950300", description="Comma-separated product codes"),
    from_country: str = Query("156", description="Origin country"),
    to_country: str = Query("everywhere", description="Destination country"),
    year_from: int = Query(2020, ge=2000, le=2024),
    year_to: int = Query(2022, ge=2000, le=2024),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=10, le=1000),
    include_tariff_data: bool = Query(False, description="Include UK tariff data for products")
):
    """Enhanced trade query that optionally includes UK tariff information."""
    
    # Get regular trade data
    trade_response = await query_trade_data(
        trade_type=trade_type,
        product_codes=product_codes,
        from_country=from_country,
        to_country=to_country,
        year_from=year_from,
        year_to=year_to,
        page=page,
        page_size=page_size
    )
    
    result = trade_response.dict()
    
    # Add tariff data if requested
    if include_tariff_data:
        product_list = [code.strip() for code in product_codes.split(",") if code.strip()]
        tariff_data = {}
        
        for product_code in product_list:
            try:
                # Pad product code to 10 digits for UK tariff system
                padded_code = product_code.ljust(10, '0')
                commodity_data = await get_commodity_details(padded_code)
                tariff_data[product_code] = commodity_data.dict()
            except Exception as e:
                # If tariff data fails, continue without it
                tariff_data[product_code] = {"error": str(e)}
        
        result["tariff_data"] = tariff_data
    
    return result

# Add new Pydantic models for detailed tariff breakdown
class DetailedTariffItem(BaseModel):
    """Detailed tariff item with duties and VAT."""
    commodity_code: str
    description: str
    vat_rate: str
    third_country_duty: str
    supplementary_unit: str

class DetailedTariffResponse(BaseModel):
    """Response for detailed tariff breakdown."""
    heading: str
    heading_description: str
    items: List[DetailedTariffItem]
    total_items: int
    execution_time_ms: float

# Add the new endpoint after your existing tariff endpoints
@app.get("/api/tariff/heading/{heading_code}/detailed", response_model=DetailedTariffResponse)
async def get_detailed_tariff_breakdown(
    heading_code: str,
    as_of: Optional[str] = Query(None, description="Date in YYYY-MM-DD format")
):
    """Get detailed tariff breakdown for a heading showing all 10-digit codes with duties and VAT."""
    start_time = datetime.now()
    
    # Validate heading code (should be 4 digits)
    if not heading_code.isdigit() or len(heading_code) != 4:
        raise HTTPException(
            status_code=400,
            detail="Heading code must be exactly 4 digits"
        )
    
    params = {}
    if as_of:
        params["as_of"] = as_of
    
    try:
        # Fetch heading data
        data = await fetch_trade_tariff_data(f"headings/{heading_code}", params)
        
        if "data" not in data:
            raise HTTPException(
                status_code=404,
                detail=f"Heading {heading_code} not found"
            )
        
        # Extract heading info
        heading_attrs = data["data"]["attributes"]
        heading_description = heading_attrs.get("formatted_description", "Unknown")
        
        # Extract commodities from included data
        commodities = []
        included_data = data.get("included", [])
        
        # Create a lookup for measure types, geographical areas, and duty expressions
        measure_types = {}
        geographical_areas = {}
        duty_expressions = {}
        
        for item in included_data:
            if item.get("type") == "measure_type":
                measure_types[item["id"]] = item.get("attributes", {}).get("description", "Unknown")
            elif item.get("type") == "geographical_area":
                geographical_areas[item["id"]] = item.get("attributes", {}).get("description", "Unknown")
            elif item.get("type") == "duty_expression":
                duty_expressions[item["id"]] = item.get("attributes", {}).get("base", "N/A")
        
        # Extract commodities and their measures
        commodity_data = {}
        
        for item in included_data:
            if item.get("type") == "commodity":
                attrs = item.get("attributes", {})
                code = attrs.get("goods_nomenclature_item_id", "")
                
                if len(code) == 10:  # Only 10-digit codes
                    commodity_data[code] = {
                        "description": attrs.get("formatted_description", "Unknown"),
                        "vat_rate": "0%",  # Default
                        "third_country_duty": "N/A",  # Default
                        "supplementary_unit": attrs.get("supplementary_unit", "N/A")
                    }
        
        # Extract measures and associate with commodities
        for item in included_data:
            if item.get("type") == "measure":
                measure_attrs = item.get("attributes", {})
                relationships = item.get("relationships", {})
                
                # Find associated commodity
                commodity_rel = relationships.get("goods_nomenclature", {}).get("data", {})
                if commodity_rel:
                    commodity_id = commodity_rel.get("id", "")
                    
                    if commodity_id in commodity_data:
                        # Get measure type
                        measure_type_rel = relationships.get("measure_type", {}).get("data", {})
                        measure_type_id = measure_type_rel.get("id", "")
                        measure_type = measure_types.get(measure_type_id, "Unknown")
                        
                        # Get geographical area
                        geo_rel = relationships.get("geographical_area", {}).get("data", {})
                        geo_id = geo_rel.get("id", "")
                        geo_area = geographical_areas.get(geo_id, "Unknown")
                        
                        # Get duty expression
                        duty_rel = relationships.get("duty_expression", {}).get("data", {})
                        duty_id = duty_rel.get("id", "")
                        duty_expr = duty_expressions.get(duty_id, "N/A")
                        
                        commodity_data[commodity_id]["measure"] = {
                            "type": measure_type,
                            "geographical_area": geo_area,
                            "duty_expression": duty_expr,
                            "effective_start_date": measure_attrs.get("effective_start_date", ""),
                            "effective_end_date": measure_attrs.get("effective_end_date")
                        }
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return DetailedTariffResponse(
            heading=heading_code,
            heading_description=heading_description,
            items=commodity_data,
            total_items=len(commodity_data),
            execution_time_ms=round(execution_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing heading {heading_code}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing heading data: {str(e)}"
        )

# Add another endpoint to get all commodities under a heading (alternative approach)
@app.get("/api/tariff/heading/{heading_code}/commodities")
async def get_heading_commodities(
    heading_code: str,
    as_of: Optional[str] = Query(None, description="Date in YYYY-MM-DD format")
):
    """Get all commodities under a heading with their measures."""
    start_time = datetime.now()
    
    # Validate and format heading code - must be exactly 4 digits, zero-padded
    if not heading_code.isdigit():
        raise HTTPException(
            status_code=400,
            detail="Heading code must contain only digits"
        )
    
    # Ensure it's exactly 4 digits, zero-padded
    if len(heading_code) > 4:
        raise HTTPException(
            status_code=400,
            detail="Heading code must be 4 digits or less"
        )
    
    # Zero-pad to 4 digits as required by the API
    heading_code = heading_code.zfill(4)
    
    params = {}
    if as_of:
        params["as_of"] = as_of
    
    try:
        # Fetch heading data with properly formatted heading_id
        heading_data = await fetch_trade_tariff_data(f"headings/{heading_code}", params)
        
        if "data" not in heading_data:
            raise HTTPException(
                status_code=404,
                detail=f"Heading {heading_code} not found"
            )
        
        included_data = heading_data.get("included", [])
        logger.info(f"Total included items for heading {heading_code}: {len(included_data)}")
        
        # Build comprehensive lookup tables
        measure_types = {}
        geographical_areas = {}
        duty_expressions = {}
        measurement_units = {}
        
        # Build lookup tables first
        for item in included_data:
            item_type = item.get("type", "")
            item_id = item.get("id", "")
            attrs = item.get("attributes", {})
            
            if item_type == "measure_type":
                measure_types[item_id] = attrs.get("description", "Unknown")
            elif item_type == "geographical_area":
                geographical_areas[item_id] = attrs.get("description", "Unknown")
            elif item_type == "duty_expression":
                duty_expressions[item_id] = attrs.get("base", "N/A")
            elif item_type == "measurement_unit":
                measurement_units[item_id] = attrs.get("description", "")
        
        logger.info(f"Built lookups: measure_types={len(measure_types)}, geo_areas={len(geographical_areas)}, duty_expr={len(duty_expressions)}, units={len(measurement_units)}")
        
        # Extract commodities with their basic info
        commodity_basic_info = {}
        
        for item in included_data:
            if item.get("type") == "commodity":
                attrs = item.get("attributes", {})
                code = attrs.get("goods_nomenclature_item_id", "")
                
                # Only include 10-digit codes that start with our heading
                if len(code) == 10 and code.startswith(heading_code):
                    commodity_basic_info[code] = {
                        "description": attrs.get("formatted_description", attrs.get("description", "Unknown")),
                        "vat_rate": "0%",
                        "third_country_duty": "N/A",
                        "supplementary_unit": "N/A"
                    }
        
        logger.info(f"Found {len(commodity_basic_info)} commodities for heading {heading_code}")
        
        # Process measures and link them to commodities
        measures_processed = 0
        measures_matched = 0
        
        for item in included_data:
            if item.get("type") == "measure":
                relationships = item.get("relationships", {})
                measure_attrs = item.get("attributes", {})
                measures_processed += 1
                
                # Find associated commodity
                commodity_code = None
                
                # Try to get commodity from relationships
                commodity_rel = relationships.get("goods_nomenclature", {}).get("data", {})
                if commodity_rel:
                    commodity_code = commodity_rel.get("id", "")
                
                # Alternative: try to get from measure attributes
                if not commodity_code:
                    commodity_code = measure_attrs.get("goods_nomenclature_item_id", "")
                
                if commodity_code and commodity_code in commodity_basic_info:
                    measures_matched += 1
                    
                    # Get measure type information
                    measure_type_id = ""
                    measure_type_rel = relationships.get("measure_type", {}).get("data", {})
                    if measure_type_rel:
                        measure_type_id = measure_type_rel.get("id", "")
                    
                    measure_type = measure_types.get(measure_type_id, f"Unknown-{measure_type_id}")
                    
                    # Get geographical area
                    geo_id = ""
                    geo_rel = relationships.get("geographical_area", {}).get("data", {})
                    if geo_rel:
                        geo_id = geo_rel.get("id", "")
                    
                    geo_area = geographical_areas.get(geo_id, f"Unknown-{geo_id}")
                    
                    # Get duty expression
                    duty_id = ""
                    duty_rel = relationships.get("duty_expression", {}).get("data", {})
                    if duty_rel:
                        duty_id = duty_rel.get("id", "")
                    
                    duty_expr = duty_expressions.get(duty_id, "N/A")
                    
                    # Get measurement unit
                    unit_id = ""
                    unit_rel = relationships.get("measurement_unit", {}).get("data", {})
                    if unit_rel:
                        unit_id = unit_rel.get("id", "")
                    
                    unit_desc = measurement_units.get(unit_id, "")
                    
                    # Enhanced logging for debugging
                    logger.info(f"Processing measure for {commodity_code}: type_id={measure_type_id}, type={measure_type}, geo_id={geo_id}, geo={geo_area}, duty_id={duty_id}, duty={duty_expr}, unit_id={unit_id}, unit={unit_desc}")
                    
                    # Categorize measures with more comprehensive matching
                    
                    # VAT measures - check multiple conditions
                    if (measure_type_id in ["305", "306", "VTS"] or 
                        "VAT" in measure_type.upper() or
                        "value added tax" in measure_type.lower() or
                        "standard rate" in measure_type.lower()):
                        commodity_basic_info[commodity_code]["vat_rate"] = duty_expr
                        logger.info(f"Set VAT for {commodity_code}: {duty_expr}")
                    
                    # Third country duty measures - enhanced matching
                    elif (measure_type_id in ["103", "105", "142", "143", "112", "115"] or  # Common third country duty IDs
                          "Third country duty" in measure_type or
                          "third country" in measure_type.lower() or
                          "autonomous tariff suspension" in measure_type.lower() or
                          "tariff preference" in measure_type.lower() or
                          "customs duty" in measure_type.lower() or
                          geo_id in ["1011", "1013", "1008"] or  # Common "Erga Omnes" and third country IDs
                          "erga omnes" in geo_area.lower() or
                          "third countries" in geo_area.lower() or
                          "all third countries" in geo_area.lower()):
                        
                        # Only update if we have a meaningful duty expression
                        if duty_expr and duty_expr != "N/A" and duty_expr.strip():
                            commodity_basic_info[commodity_code]["third_country_duty"] = duty_expr
                            logger.info(f"Set third country duty for {commodity_code}: {duty_expr}")
                    
                    # Supplementary unit - enhanced extraction
                    if unit_desc and unit_desc.strip() and unit_desc != "N/A":
                        commodity_basic_info[commodity_code]["supplementary_unit"] = unit_desc
                        logger.info(f"Set supplementary unit for {commodity_code}: {unit_desc}")
                    elif measure_attrs.get("supplementary_unit"):
                        supp_unit = measure_attrs.get("supplementary_unit")
                        commodity_basic_info[commodity_code]["supplementary_unit"] = supp_unit
                        logger.info(f"Set supplementary unit from attrs for {commodity_code}: {supp_unit}")
        
        logger.info(f"Processed {measures_processed} measures, matched {measures_matched} to commodities")
        
        # Convert to final format
        detailed_commodities = []
        for code, info in commodity_basic_info.items():
            detailed_commodities.append({
                "commodity_code": code,
                "description": info["description"],
                "vat_rate": info["vat_rate"],
                "third_country_duty": info["third_country_duty"],
                "supplementary_unit": info["supplementary_unit"]
            })
        
        # Sort by commodity code
        detailed_commodities.sort(key=lambda x: x["commodity_code"])
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "heading": heading_code,
            "heading_description": heading_data["data"]["attributes"].get("formatted_description", "Unknown"),
            "commodities": detailed_commodities,
            "total_commodities": len(detailed_commodities),
            "execution_time_ms": round(execution_time, 2),
            "debug_info": {
                "measures_processed": measures_processed,
                "measures_matched": measures_matched,
                "lookup_tables_built": {
                    "measure_types": len(measure_types),
                    "geographical_areas": len(geographical_areas),
                    "duty_expressions": len(duty_expressions),
                    "measurement_units": len(measurement_units)
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing heading {heading_code}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing heading data: {str(e)}"
        )

# Update the debug endpoint to use proper heading formatting
@app.get("/api/tariff/heading/{heading_code}/debug")
async def debug_heading_data(heading_code: str):
    """Debug endpoint to see raw heading data structure."""
    try:
        # Ensure proper formatting
        heading_code = heading_code.zfill(4)
        
        data = await fetch_trade_tariff_data(f"headings/{heading_code}")
        
        # Extract and categorize all included items for debugging
        debug_info = {
            "heading_code": heading_code,
            "heading_data": data.get("data", {}),
            "included_summary": {},
            "measure_types": {},
            "geographical_areas": {},
            "duty_expressions": {},
            "measurement_units": {},
            "sample_measures": [],
            "sample_commodities": []
        }
        
        included_data = data.get("included", [])
        
        # Count types and build lookups
        for item in included_data:
            item_type = item.get("type", "unknown")
            debug_info["included_summary"][item_type] = debug_info["included_summary"].get(item_type, 0) + 1
            
            if item_type == "measure_type":
                debug_info["measure_types"][item["id"]] = item.get("attributes", {}).get("description", "")
            elif item_type == "geographical_area":
                debug_info["geographical_areas"][item["id"]] = item.get("attributes", {}).get("description", "")
            elif item_type == "duty_expression":
                debug_info["duty_expressions"][item["id"]] = item.get("attributes", {}).get("base", "")
            elif item_type == "measurement_unit":
                debug_info["measurement_units"][item["id"]] = item.get("attributes", {}).get("description", "")
            elif item_type == "measure" and len(debug_info["sample_measures"]) < 3:
                debug_info["sample_measures"].append({
                    "id": item.get("id"),
                    "attributes": item.get("attributes", {}),
                    "relationships": item.get("relationships", {})
                })
            elif item_type == "commodity" and len(debug_info["sample_commodities"]) < 3:
                debug_info["sample_commodities"].append({
                    "id": item.get("id"),
                    "attributes": item.get("attributes", {}),
                    "relationships": item.get("relationships", {})
                })
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e)}

# Add this simplified debug endpoint to see exactly what we're getting
@app.get("/api/tariff/heading/{heading_code}/raw")
async def get_raw_heading_data(heading_code: str):
    """Get raw heading data to understand the structure."""
    try:
        data = await fetch_trade_tariff_data(f"headings/{heading_code}")
        
        # Return just the first few items of each type for inspection
        result = {
            "main_data": data.get("data", {}),
            "included_count": len(data.get("included", [])),
            "included_types": {},
            "sample_commodity": None,
            "sample_measure": None,
            "sample_measure_type": None,
            "sample_duty_expression": None,
            "sample_geographical_area": None
        }
        
        included_data = data.get("included", [])
        
        for item in included_data:
            item_type = item.get("type", "unknown")
            result["included_types"][item_type] = result["included_types"].get(item_type, 0) + 1
            
            # Get samples
            if item_type == "commodity" and not result["sample_commodity"]:
                result["sample_commodity"] = item
            elif item_type == "measure" and not result["sample_measure"]:
                result["sample_measure"] = item
            elif item_type == "measure_type" and not result["sample_measure_type"]:
                result["sample_measure_type"] = item
            elif item_type == "duty_expression" and not result["sample_duty_expression"]:
                result["sample_duty_expression"] = item
            elif item_type == "geographical_area" and not result["sample_geographical_area"]:
                result["sample_geographical_area"] = item
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

