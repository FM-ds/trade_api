from fastapi import FastAPI, Query
from typing import List, Optional
from sqlalchemy import create_engine, text
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec


import os
import uvicorn
import json

with open("creds.json", "r") as f:
    creds = json.load(f)
    pc = Pinecone(
        api_key=creds["pinecone"]
    )
    DB_URL = creds["db"]


product_codes_index = pc.Index("product-codes-index")
country_index = pc.Index("country-codes-index")

app = FastAPI(title="Trade Data API")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

### 1️⃣ Exports of product_code X from country A to all others ###
@app.get("/exports/")
def get_exports(
    product_code: int,
    exporter: int,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    metric: str = Query("value", enum=["value", "quantity"])
):
    """
    Returns exports of product_code X from country A to all others.
    Defaults to the most recent year if no range is provided.
    """
    with engine.connect() as conn:
        sql = f"""
        SELECT year, importer, {metric}
        FROM trade_table
        WHERE product_code = :product_code AND exporter = :exporter
        """
        params = {"product_code": product_code, "exporter": exporter}

        if start_year and end_year:
            sql += " AND year BETWEEN :start_year AND :end_year"
            params["start_year"] = start_year
            params["end_year"] = end_year
        else:
            sql += " ORDER BY year DESC LIMIT 1"

        print("foo")
        result = conn.execute(text(sql), params).mappings().all()
    print(result)
    return {"exports": list(result)}


@app.get("/test/")
def get_test():
    return {"test": "test"}

### 2️⃣ Total imports/exports of product_code X for country A by year ###
@app.get("/trade_totals/")
def get_trade_totals(
    product_code: int,
    country: int,
    trade_type: str = Query("export", enum=["export", "import"]),
    metric: str = Query("value", enum=["value", "quantity"])
):
    """
    Returns total imports or exports of product_code X for country A by year.
    """
    column = "exporter" if trade_type == "export" else "importer"

    with engine.connect() as conn:
        sql = f"""
        SELECT year, SUM({metric}) AS total_{metric}
        FROM trade_table
        WHERE product_code = :product_code AND {column} = :country
        GROUP BY year
        ORDER BY year DESC
        """
        result = conn.execute(text(sql), {"product_code": product_code, "country": country}).fetchall()
        return {"trade_totals": [dict(row) for row in result]}

### 3️⃣ Top exports/imports in year A for country X ###
@app.get("/top_trade/")
def get_top_trade(
    year: int,
    country: int,
    trade_type: str = Query("export", enum=["export", "import"]),
    metric: str = Query("value", enum=["value", "quantity"]),
    limit: int = 10
):
    """
    Returns top exports/imports for country X in year A.
    """
    column = "exporter" if trade_type == "export" else "importer"

    with engine.connect() as conn:
        sql = f"""
        SELECT product_code, SUM({metric}) AS total_{metric}
        FROM trade_table
        WHERE {column} = :country AND year = :year
        GROUP BY product_code
        ORDER BY total_{metric} DESC
        LIMIT :limit
        """
        result = conn.execute(text(sql), {"year": year, "country": country, "limit": limit}).fetchall()
        return {"top_trade": [dict(row) for row in result]}
    
if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)

@app.get("/top_traders/")
def get_top_traders(
    product_code: int,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    trade_type: str = Query("export", enum=["export", "import"]),
    filter_country: Optional[int] = None,  # Filter for a specific exporter/importer
    metric: str = Query("value", enum=["value", "quantity"]),
    num_countries: int = 5,  # Default: Show the top 5 traders
    limit: int = 100  # Default: Limit the final results to 100
):
    """
    Returns the top traders (exporters or importers) of a given product_code within a date range.
    - Now includes `country_name`, `country_iso2`, and `country_iso3`.
    """

    column = "exporter" if trade_type == "export" else "importer"
    filter_column = "importer" if trade_type == "export" else "exporter"

    with engine.connect() as conn:
        # Step 1: Identify the top `num_countries` traders from the last year in the range
        sql_top_countries = f"""
        SELECT t.{column} AS trader, SUM(t.{metric}) AS total_{metric}, c.country_name, c.country_iso2, c.country_iso3
        FROM trade_table t
        LEFT JOIN country_codes c ON t.{column} = c.country_code
        WHERE t.product_code = :product_code
          AND t.year = (SELECT MAX(year) FROM trade_table WHERE product_code = :product_code)
        """
        params = {"product_code": product_code}

        if bool(filter_country):
            sql_top_countries += f" AND t.{filter_column} = :filter_country"
            params["filter_country"] = filter_country

        sql_top_countries += f"""
        GROUP BY t.{column}, c.country_name, c.country_iso2, c.country_iso3
        ORDER BY total_{metric} DESC
        LIMIT :num_countries
        """
        params["num_countries"] = num_countries

        top_traders = conn.execute(text(sql_top_countries), params).mappings().all()
        top_trader_ids = [t["trader"] for t in top_traders]

        if not top_trader_ids:
            return {"top_traders": []}  # No data found

        # Step 2: Get time series data for the top traders over the full range
        sql_timeseries = f"""
        SELECT t.{column} AS trader, t.year, SUM(t.{metric}) AS total_{metric}, c.country_name, c.country_iso2, c.country_iso3
        FROM trade_table t
        LEFT JOIN country_codes c ON t.{column} = c.country_code
        WHERE t.product_code = :product_code
          AND t.{column} IN :top_trader_ids
        """
        params["top_trader_ids"] = tuple(top_trader_ids)  # Pass as a tuple for SQL query

        if start_year and end_year:
            sql_timeseries += " AND t.year BETWEEN :start_year AND :end_year"
            params["start_year"] = start_year
            params["end_year"] = end_year
        elif start_year:
            sql_timeseries += " AND t.year = :start_year"
            params["start_year"] = start_year

        sql_timeseries += f"""
        GROUP BY t.{column}, t.year, c.country_name, c.country_iso2, c.country_iso3
        ORDER BY t.year DESC, total_{metric} DESC
        LIMIT :limit
        """
        params["limit"] = limit

        result = conn.execute(text(sql_timeseries), params).mappings().all()
        return {"data": list(result)}
    

@app.get("/autocomplete")
def autocomplete_search(query: str, type: str = Query("product", enum=["product", "country"]), top_k: int = 5):
    """
    Returns autocomplete suggestions based on input query.
    """
    index = product_codes_index if type == "product" else country_index

    if query.isdigit():
        result = index.fetch([query])
        if result.vectors:
            return [{"code": query, "description": result.vectors[query].metadata["description"]}]
        else:
            return [{"message": f"{type.capitalize()} code not found"}]

    # Convert query to embeddings
    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"}
    )[0].values

    # Search Pinecone index
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Format results
    matches = [{"code": match.id, "description": match.metadata["description"]}
               for match in results.matches]

    return matches

# For the bar chart: For a importer/exporter in a given year, and product code, show the top 5 countries 

@app.get("/top_countries/")
def get_top_countries(
    year: int,
    country: int,
    product_code: int,
    trade_type: str = Query("exporter", enum=["exporter", "importer"]),
    metric: str = Query("value", enum=["value", "quantity"]),
    limit: int = 5
):
    """
    Returns the top countries for a given importer/exporter in a given year and product code.
    """

    trade_partner_col = "importer" if trade_type == "exporter" else "exporter"

    with engine.connect() as conn:
        sql = f"""
            SELECT t.*, c.country_name, c.country_iso2, c.country_iso3
            FROM trade_table t
            LEFT JOIN country_codes c ON t.{trade_partner_col} = c.country_code
            WHERE 
                t.year = :year AND 
                t.{trade_type} = :country AND 
                t.product_code = :product_code   
            ORDER BY t.{metric} DESC 
            LIMIT :limit
        """
        params = {
            "year": year,
            "country": country,
            "product_code": product_code,
            "limit": limit
        }
        result = conn.execute(text(sql), params).mappings().all()

    return {"data": list(result)}



@app.get("/top_countries_timeseries/")
def get_top_countries(
    start_year: int,
    end_year: int,
    importer: int,
    exporter: int,
    product_code: int,
    trade_type: str = Query("exporter", enum=["exporter", "importer"]),
    metric: str = Query("value", enum=["value", "quantity"]),
    limit: int = 5
):
    """
    Returns the top countries for a given importer/exporter in a given year and product code.
    """

    trade_partner_col = "importer" if trade_type == "exporter" else "exporter"

    with engine.connect() as conn:
        # sql = f"""
        #     SELECT t.*, c.country_name, c.country_iso2, c.country_iso3 
        #     FROM trade_table t
        #     LEFT JOIN country_codes c ON t.importer = c.country_code
        #     WHERE 
        #         t.year BETWEEN :start_year AND :end_year AND
        #         t.exporter = :exporter AND
        #         t.importer = :importer AND
        #         t.product_code = :product_code   
        # """

        sql = """
                SELECT 
                    t.*, 
                    exp.country_name AS exporter_name, exp.country_iso2 AS exporter_iso2, exp.country_iso3 AS exporter_iso3,
                    imp.country_name AS importer_name, imp.country_iso2 AS importer_iso2, imp.country_iso3 AS importer_iso3
                FROM trade_table t
                LEFT JOIN country_codes exp ON t.exporter = exp.country_code
                LEFT JOIN country_codes imp ON t.importer = imp.country_code
                WHERE 
                    t.year BETWEEN :start_year AND :end_year AND
                    t.exporter = :exporter AND
                    t.importer = :importer AND
                    t.product_code = :product_code   
            """

        params = {
            "start_year": start_year,
            "end_year": end_year,
            "exporter": exporter,
            "importer": importer,
            "product_code": product_code,
            # "limit": limit
        }
        result = conn.execute(text(sql), params).mappings().all()

    return {"data": list(result)}

@app.get("/partners_for_product_in_year/")
def partners_for_product_in_year(
    year: int,
    country: int,
    product_code: int,
    trade_type: str = Query("exporter", enum=["exporter", "importer"]),
    metric: str = Query("value", enum=["value", "quantity"])
):
    # Determine the relevant column for trade partners
    trade_partner_col = "importer" if trade_type == "exporter" else "exporter"

    with engine.connect() as conn:
        sql = f"""
            SELECT 
                t.{trade_partner_col} AS partner_id,
                c.country_name AS partner_name,
                c.country_iso3 AS partner_iso3,
                t.{metric} AS {metric}
            FROM trade_table t
            LEFT JOIN country_codes c ON t.{trade_partner_col} = c.country_code
            WHERE 
                t.year = :year AND
                t.{trade_type} = :country AND
                t.product_code = :product_code
            ORDER BY t.{metric} DESC
        """
        params = {
            "year": year,
            "country": country,
            "product_code": product_code
        }
        result = conn.execute(text(sql), params).mappings().all()
    
    return result

@app.get("/country_name_from_code/")
def country_name_from_code(country_code: int):
    with engine.connect() as conn:
        sql = """
            SELECT country_name
            FROM country_codes
            WHERE country_code = :country_code
        """
        params = {"country_code": country_code}
        result = conn.execute(text(sql), params).first()
    
    return result

@app.get("/product_name_from_code/")
def product_name_from_code(product_code: int):
    with engine.connect() as conn:
        sql = """
            SELECT description
            FROM product_codes
            WHERE code = :product_code
        """
        params = {"product_code": product_code}
        result = conn.execute(text(sql), params).first()
    
    return result