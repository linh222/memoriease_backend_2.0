# MemoriEase FastAPI Backend

## Overview

This repository is for building a fastAPI backend for the MemoriEase system. The system use LAVIS to run the BLIP-2 
embedding model and fastAPI to serve several endpoints.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Endpoints](#endpoints)
- [Testing](#testing)

## Installation

Describe how to install your FastAPI project. Include any dependencies or prerequisites that users need to install. You can provide a step-by-step guide or use code blocks if necessary.

```bash
# Clone the repository
git clone git@github.com:linh222/memoriease_backend_2.0.git

# Change into the project directory
cd memoriease_backend_2.0

# Create a virtual environment (optional but recommended)
python -m venv memoriease_backend_2

# Activate the virtual environment
source memoriease_backend_2/bin/activate  # On Windows, use `memoriease_backend_2\Scripts\activate`

# Install project dependencies
pip install -r requirements.txt
 ```

## Configuration
Because the system uses Elaticsearch as the database for saving and retrieving data, install the ELastic search from the
source: https://www.elastic.co/guide/en/welcome-to-elastic/8.6/index.html (Recommended version: 8.6)

Ingest the data, get the data from the owner.
```bash
# Run elastic search
cd folder/to/elasticsearch
./bin/elasticsearch

# Ingest data
cd memoriease_backend_2.0
python app/download_nltk.py
python app/ingest_data_lsc24.py
```

## Endpoints
List and describe the API endpoints provided by your FastAPI application. Include details such as endpoint URLs, 
request methods, and expected responses.
+ /: GET get the main root, the model name
+ /info: GET the information of the model
+ /health: GET check the health status
+ /image: POST get the image by date, with following parameter
  + day_month_year: format yyyy-mm-dd, ex: 2019-01-01
  + time_period: (optional) morning or afternoon
  + hour: (optional) any hour from 0 to 23
+ /metadata: POST for frontend to submit the results after submission for logging.
+ /predict: POST to predict for lsc24, with following parameter
  + query: full text query, that can be auto filter
  + semantic_name: the semantic name for filtering
  + Advanced filters: @weekend:1/0, @start:0->23, @end:0-23, @ocr:text, @location:text
+ /predict_temporal: POST to predict in temporal with previous and next event, with following parameters:
  + query: the current event query
  + semantic_name: the filter by semantic name
  + previous_event: the query for previous event
  + next_event: the query for next event
  + time_gap: the time gap in hour
+ /visual_similarity: POST retrieve images by visual
  + query: the full-text query to initial retrieve and filters
  + image_id: the list of relevant images
+ /conversational_search: POST request to ask and answer by conversational search lifelog
  + query: the current chat of users
  + previous_chat: list or previous chat from users


## Testing

Run the fastapi in local
```./local_server.sh```

Access to http://localhost:8080/docs to test the endpoints.
