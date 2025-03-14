# TradeLens

![TradeLens Screenshot](/docs/imgs/screenshot.jpeg)


TradeLens lets analysts quickly request and visualise trade flow data. Users search for products and countries, and receive AI semantically-aware search results, and the tool fetches data and presents it with interactive charts. 

## Architecture

![TradeLens stack](/docs/imgs/stack.jpeg)

The MVP tool, built during the Hackathon, employed three services, two hosted on Azure.
Trade flows database. An Azure SQL Database holds cleaned and transformed trade flow data, sourced from [NAME OF DATA SOURCE]. The database is populated by a data ingestion Python script. The annual release schedule of the source data means ingestion is infrequent.
Products vector database. A PineconeDB database holds embeddings for each HS item, and country name. This facilitates the semantic search of products and locations. This should be replaced by its Azure equivalent, Azure AI Search. 
Web app. A FastAPI server via Azure’s Web App Service hosts two components:
Frontend. Serves a basic UI for the querying and Vega-lite driven visualisation of trade data.
Backend. Serves a simple API, which the frontend invokes to fetch data from the trade flows database.

## User flow

To see how these components join together, consider a simple user flow: a user requests the top export destinations for Kazakhstani Uranium.
User searches for products and countries. The user enters “Uranium” and “Kazakhstan” in the product and country search boxes.
Autocompletion. Frontend Javascript recognises text that has been entered, and invokes the backend API’s /autocomplete endpoint to fetch matching products and countries.

To provide the suggestions, the backend requests an embedding for the user’s search term, and queries the PineconeDB with this embedding to find products which are the nearest neighbours to the users’ search.
User submits request. Happy with the suggested product and country suggestions, the user clicks a “Fetch data” button. 
Data retrieval. Frontend Javascript responds to the button press by invoking the partners_for_product_in_year API endpoint. The API queries the SQL database for the requested data, and returns it.
Data visualization. Frontend Javascript uses the returned data to create a simple Vega-lite chart specification which shows the requested data. 

## Testing

The tool was served via Azure’s Web App Service. It relies on a SQL database for the trade flows and a PineconeDB database for the semantic search.

Credentials are excluded from this repository. The code references a `creds.json` file - a quick Hackathon bodge that should be replaced with Azure Key Vault or similar.

This file is structured as follows:

```json
{
    "db": "mysql+pymysql://[...]",
    "pinecone": "pcsk_[...]"
}```
