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
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

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
python app/ingest_data_no_segmentation.py
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
+ /predict: POST to predict for lsc23, with following parameter
  + query: full text query, that can be auto filter
  + topic: topic name for logging
  + semantic_name: the semantic name for filtering
  + start_hour: (optional) 0-> 23 for filtering start hour of event
  + end_hour: (optional) 0-> 23 for filtering end hour of event
  + is_weekend: (optional) 1 or 0, filter the event is in weekend or not.
+ /predict_temporal: POST to predict in temporal with previous and next event, with following parameters:
  + query: the current event query
  + semantic_name: the filter by semantic name
  + previous_event: the query for previous event
  + next_event: the query for next event
  + time_gap: the time gap in hour
+ /relevance_feedback: POST retrieve images by positive relevant feedback
  + query: the full-text query
  + image_id: the list of relevant images
  + semantic_name: the semantic name for filtering


## Testing

Run the fastapi in local
```./local_server.sh```

Access to http://localhost:8080/docs to test the endpoints.

