import os
import requests

from mistralai import Mistral
import json

from src.text2sparql.entity_linking import TEXT2JSON2SPARQLPipeline

from dotenv import load_dotenv
import os
load_dotenv()
SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT")
SPARQL_ENDPOINT_KEY = os.getenv("SPARQL_ENDPOINT_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_AGENT_ID = os.getenv("MISTRAL_AGENT_ID")

class TEXT2SPARQLRetrieval:

    headers = {
    'accept': 'application/json',
    'X-API-Key': SPARQL_ENDPOINT_KEY,
    'Content-Type': 'application/sparql-query',
    }
    sparql_endpoint = SPARQL_ENDPOINT 

    def __init__(self):
        self.pipeline = TEXT2JSON2SPARQLPipeline()
        self.client = Mistral(api_key=MISTRAL_API_KEY)

    def text2json(self, text_input):
        response = self.client.beta.conversations.start_stream(
            agent_id=MISTRAL_AGENT_ID,
            inputs=text_input,
        )

        json_text = ""
        for chunk in response:
            try: json_text+=chunk.data.content
            except: pass
        if json_text.startswith("```json") and json_text.endswith("```"):
            return json.loads(json_text[7:-3])
        else:
            return json.loads(json_text)

    def json2sparql(self, json_input):
        processed_json, sparql = self.pipeline.process_json_to_sparql(json_input)
        return sparql
    
    def run_sparql(self, sparql_query):
        response = requests.post(self.sparql_endpoint, headers=self.headers, data=sparql_query)

        response.raise_for_status()
        
        data = response.json()
        bindings = data.get("results", {}).get("bindings", [])

        artefacts = [r['DigitalArtefact']['value'].split('/')[-1] for r in bindings]
        return artefacts
    
    def retrieval(self, query_input):
        print("Processing query:", query_input)
        json_input = self.text2json(query_input)
        sparql_query = self.json2sparql(json_input)
        results = self.run_sparql(sparql_query)
        print("SPARQL Results:", len(results))
        return results

