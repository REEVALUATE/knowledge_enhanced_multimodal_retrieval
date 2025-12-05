"""
Sparnatural JSON from LLM Post-Processing with Reconciliation
"""

import requests
from typing import Dict, List, Any, Optional, Set, Tuple
from urllib.parse import quote
import json
import time
from dataclasses import dataclass
from src.text2sparql.json2sparql import SparnaturalToSparql
from SPARQLWrapper import SPARQLWrapper, JSON

# load env variables from .env file
from dotenv import load_dotenv
import os
load_dotenv()
SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT")
SPARQL_ENDPOINT_KEY = os.getenv("SPARQL_ENDPOINT_KEY")


import re

# Compile regex patterns once for better performance
_VALUE_PATTERN = re.compile(r'\?Value_(\d+)')
_SUBJECT_PATTERN = re.compile(r'\?(\w+)\s+<[^>]*P43_has_dimension[^>]*>\s+\?Dimension_\d+')
_PAINTING_PATTERN = re.compile(r'\?(\w*Painting\w*)')
_P43_PATTERN = re.compile(r'\s*\?\w+\s+<[^>]*P43_has_dimension[^>]*>\s+\?Dimension_\d+\s*\.')
_E54_PATTERN = re.compile(r'\s*\?Dimension_\d+\s+rdf:type\s+<[^>]*E54_Dimension[^>]*>\s*\.')
_OLD_P90_PATTERN = re.compile(r'\s*\?Dimension_\d+\s+<[^>]*P90_has_value[^>]*>\s+\?Value_\d+\s*\.')
_WHERE_PATTERN = re.compile(r'WHERE\s*\{', re.IGNORECASE)
_LABEL_PATTERN = re.compile(r"(\?[A-Za-z_][A-Za-z0-9_]*)\s+<http://www\.w3\.org/2000/01/rdf-schema#label>\s+(\?[A-Za-z_][A-Za-z0-9_]*)\s*\.")

def fix_dimension_query(sparql: str) -> str:
    """
    自动修复SPARQL中的维度查询。
    
    检测有多少个 P90_has_value，然后：
    1. 为每个值创建独立的 ?Dimension_N 变量
    2. 添加对应的 P43_has_dimension 三元组
    3. 更新 E54_Dimension 类型声明
    
    Args:
        sparql: 原始SPARQL查询字符串
        
    Returns:
        修复后的SPARQL查询字符串
    """
    
    # 1. 检测有多少个 P90_has_value (计算 ?Value_N 的数量) - 使用预编译的 pattern
    value_matches = _VALUE_PATTERN.findall(sparql)
    
    if not value_matches:
        return sparql
    
    # 获取所有 Value 的编号并排序
    value_numbers = sorted(set(int(n) for n in value_matches))
    num_dimensions = len(value_numbers)

    
    # 2. 查找主体变量（通常是 ?Painting_1 或类似的）- 使用预编译的 pattern
    subject_match = _SUBJECT_PATTERN.search(sparql)
    
    if subject_match:
        subject_var = f"?{subject_match.group(1)}"
    else:
        # 尝试从其他地方推断主体变量
        painting_match = _PAINTING_PATTERN.search(sparql)
        subject_var = f"?{painting_match.group(1)}" if painting_match else "?Painting_1"
    
    # 3. 移除旧的维度相关三元组 - 使用预编译的 patterns
    # 移除旧的 P43_has_dimension
    p43_escape_pattern = re.compile(
        r'\s*' + re.escape(subject_var) + r'\s+<[^>]*P43_has_dimension[^>]*>\s+\?Dimension_\d+\s*\.'
    )
    sparql = p43_escape_pattern.sub('', sparql)
    
    # 移除旧的 E54_Dimension 类型声明
    sparql = _E54_PATTERN.sub('', sparql)
    
    # 移除旧的 P90_has_value (我们会重新生成)
    sparql = _OLD_P90_PATTERN.sub('', sparql)
    
    # 4. 在查询主体开始位置插入新的维度三元组
    # 找到主体变量第一次出现的位置（通常在 WHERE { 之后）- 使用预编译的 pattern
    where_match = _WHERE_PATTERN.search(sparql)
    
    if not where_match:
        print("❌ 未找到 WHERE 子句")
        return sparql
    
    insert_pos = where_match.end()
    
    # 构建新的维度三元组
    new_triples = []
    
    # 添加注释
    new_triples.append("\n  # Dimensions (auto-fixed)")
    
    # 为每个维度生成三元组
    for i, value_num in enumerate(value_numbers, 1):
        dim_var = f"?Dimension_{i}"
        value_var = f"?Value_{value_num}"
        
        # P43_has_dimension
        new_triples.append(
            f"\n  {subject_var} <http://www.cidoc-crm.org/cidoc-crm/P43_has_dimension> {dim_var}."
        )
        
        # E54_Dimension type
        new_triples.append(
            f"\n  {dim_var} rdf:type <http://www.cidoc-crm.org/cidoc-crm/E54_Dimension>."
        )
        
        # P90_has_value
        new_triples.append(
            f"\n  {dim_var} <http://www.cidoc-crm.org/cidoc-crm/P90_has_value> {value_var}."
        )
    
    # 插入新的三元组
    new_triples_str = ''.join(new_triples)
    sparql = sparql[:insert_pos] + new_triples_str + sparql[insert_pos:]
    
    # print(f"✓ 已生成 {num_dimensions} 个独立的维度变量")
    
    return sparql

@dataclass
class QueryInput:
    """Reconciliation"""
    query: str
    type: Optional[str] = None
    predicate: Optional[str] = None


@dataclass
class ReconciliationResult:
    """Reconciliation"""
    id: str
    name: str

# ============================================
# Reconciliation Service
# ============================================

class ReconciliationService:
    """SPARQL Reconciliation"""
    headers = {
    'accept': 'application/json',
    'X-API-Key': SPARQL_ENDPOINT_KEY,
    'Content-Type': 'application/sparql-query',
    }
    sparql_endpoint = SPARQL_ENDPOINT 
    def __init__(self, max_results: int = 10):

        self.MAX_RESULTS = max_results
        self.wikidata_query = SPARQLWrapper("https://query.wikidata.org/sparql")
        # Cache for reconciliation results
        self._cache = {}
    
    def _execute_sparql(self, query: str) -> List[str]:
        """SPARQL"""
        try:

            response = requests.post(self.sparql_endpoint, headers=self.headers, data=query)

            response.raise_for_status()
            
            data = response.json()
            bindings = data.get("results", {}).get("bindings", [])
            
            return [b["x"]["value"] for b in bindings if "x" in b]
        
        except Exception as e:
            print(f"❌ SPARQL ERROR: {e}")
            return []
    
    def _format_results(self, uri_list: List[str], name: str) -> List[ReconciliationResult]:

        sorted_uris = sorted(uri_list, key=len)
        
        results = []
        for i, uri in enumerate(sorted_uris):
            results.append(ReconciliationResult(
                id=uri,
                name=name
            ))
        
        return results
    
    def search_entity(
        self, 
        name: str, 
        type_uri: Optional[str] = None,
        predicate: Optional[str] = None
    ) -> List[ReconciliationResult]:
        """
        Search nameindividual entity by name and optional type URIs
        """

        escaped_name = name.replace('"', '\\"').lower()

        type_filter = ""
        if type_uri:
            if len(type_uri) == 1:
                type_filter = f"?x a <{type_uri[0]}> ."
            else:
                optional_parts = "\n".join(
                    [f"OPTIONAL {{ ?x a <{t}> . }}" for t in type_uri]
                )
                type_filter = f"""
                {{
                {optional_parts}
                FILTER({" || ".join([f"EXISTS {{ ?x a <{t}> }}" for t in type_uri])})
                }}
                """
        if predicate != "http://www.cidoc-crm.org/cidoc-crm/P62_depicts":
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT DISTINCT ?x WHERE {{
            {type_filter}
            {{
                ?s <{predicate}> ?x .
                ?x rdfs:label ?label .
            }}
            UNION
            {{
                ?external skos:exactMatch ?x .
                ?external rdfs:label ?label .
            }}
                FILTER(
                LCASE(STR(?label)) = "{escaped_name}" ||
                STRSTARTS(LCASE(?label), "{escaped_name}") ||
                STRENDS(LCASE(?label), "{escaped_name}") ||
                CONTAINS(LCASE(?label), "{escaped_name}") ||
                STRSTARTS("{escaped_name}", LCASE(?label)) ||
                STRENDS("{escaped_name}", LCASE(?label)) ||
                CONTAINS("{escaped_name}", LCASE(?label))
                )
            }}
            """
        else:
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT DISTINCT ?x WHERE {{
            {type_filter}
            {{
                ?s <{predicate}> ?x .
                ?x rdfs:label ?label .
            }}
            UNION
            {{
                ?external skos:exactMatch ?x .
                ?external rdfs:label ?label .
            }}
                FILTER(
                LCASE(STR(?label)) = "{escaped_name}"
                )
            }}
            """
        uris = self._execute_sparql(query)
        uris = list(set(uris))  
        results = self._format_results(uris, name)

        
        return results
    
    def reconcile_batch(
        self, 
        queries: Dict[str, QueryInput]
    ) -> Dict[str, List[ReconciliationResult]]:
        """
        Batch reconciliation with caching and batched SPARQL queries
        """
        results = {}
        
        # Step 1: Check cache first
        uncached_queries = {}
        for key, query_input in queries.items():
            cache_key = (query_input.query.lower(), str(query_input.type), query_input.predicate)
            if cache_key in self._cache:
                results[key] = self._cache[cache_key]
            else:
                uncached_queries[key] = query_input
        
        if not uncached_queries:
            return results  # All from cache
        
        # Step 2: Group queries by (type, predicate) for batch processing
        grouped_queries = {}
        for key, query_input in uncached_queries.items():
            group_key = (str(query_input.type), query_input.predicate)
            if group_key not in grouped_queries:
                grouped_queries[group_key] = []
            grouped_queries[group_key].append((key, query_input))
        
        # Step 3: Execute batched queries
        for (type_uri, predicate), queries_list in grouped_queries.items():
            # Build batch query
            batch_results = self._search_entity_batch(
                [(q[1].query, q[1].type, q[1].predicate) for q in queries_list]
            )
            
            # Distribute results and cache
            for (key, query_input), entity_results in zip(queries_list, batch_results):
                results[key] = entity_results
                cache_key = (query_input.query.lower(), str(query_input.type), query_input.predicate)
                self._cache[cache_key] = entity_results
        
        return results
    
    def _search_entity_batch(
        self,
        queries: List[Tuple[str, Optional[List[str]], str]]
    ) -> List[List[ReconciliationResult]]:
        """
        Batch search for multiple entities in a single SPARQL query
        Returns results in the same order as input queries
        """
        if not queries:
            return []
        
        # Extract unique types and predicates
        type_uri = queries[0][1] if queries[0][1] else None
        predicate = queries[0][2]
        
        # Build VALUES clause for batch querying
        escaped_names = [q[0].replace('"', '\\"').lower() for q in queries]
        
        # Build type filter (same as before)
        type_filter = ""
        if type_uri:
            if len(type_uri) == 1:
                type_filter = f"?x a <{type_uri[0]}> ."
            else:
                optional_parts = "\n".join(
                    [f"OPTIONAL {{ ?x a <{t}> . }}" for t in type_uri]
                )
                type_filter = f"""
                {{
                {optional_parts}
                FILTER({" || ".join([f"EXISTS {{ ?x a <{t}> }}" for t in type_uri])})
                }}
                """
        
        # Build unified query for all names
        filter_conditions = []
        for escaped_name in escaped_names:
            filter_conditions.append(f"""
                (LCASE(STR(?label)) = "{escaped_name}" ||
                STRSTARTS(LCASE(?label), "{escaped_name}") ||
                STRENDS(LCASE(?label), "{escaped_name}") ||
                CONTAINS(LCASE(?label), "{escaped_name}") ||
                STRSTARTS("{escaped_name}", LCASE(?label)) ||
                STRENDS("{escaped_name}", LCASE(?label)) ||
                CONTAINS("{escaped_name}", LCASE(?label)))""")
        
        combined_filter = " || ".join(filter_conditions)
        
        if predicate != "http://www.cidoc-crm.org/cidoc-crm/P62_depicts":
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT DISTINCT ?x ?label WHERE {{
            {type_filter}
            {{
                ?s <{predicate}> ?x .
                ?x rdfs:label ?label .
            }}
            UNION
            {{
                ?external skos:exactMatch ?x .
                ?external rdfs:label ?label .
            }}
                FILTER({combined_filter})
            }}
            """
        else:
            # More strict filter for P62_depicts
            exact_filter = " || ".join([f'LCASE(STR(?label)) = "{name}"' for name in escaped_names])
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT DISTINCT ?x ?label WHERE {{
            {type_filter}
            {{
                ?s <{predicate}> ?x .
                ?x rdfs:label ?label .
            }}
            UNION
            {{
                ?external skos:exactMatch ?x .
                ?external rdfs:label ?label .
            }}
                FILTER({exact_filter})
            }}
            """
        
        # Execute query
        try:
            response = requests.post(self.sparql_endpoint, headers=self.headers, data=query)
            response.raise_for_status()
            data = response.json()
            bindings = data.get("results", {}).get("bindings", [])
            
            # Group results by matched label
            results_by_label = {}
            for b in bindings:
                if "x" in b and "label" in b:
                    uri = b["x"]["value"]
                    label = b["label"]["value"].lower()
                    if label not in results_by_label:
                        results_by_label[label] = []
                    results_by_label[label].append(uri)
            
            # Match results back to original queries
            batch_results = []
            for query_name, _, _ in queries:
                query_name_lower = query_name.lower()
                matched_uris = []
                
                # Find matching URIs
                for label, uris in results_by_label.items():
                    # Check if this label matches the query
                    if (query_name_lower == label or
                        label.startswith(query_name_lower) or
                        label.endswith(query_name_lower) or
                        query_name_lower in label or
                        query_name_lower.startswith(label) or
                        query_name_lower.endswith(label) or
                        label in query_name_lower):
                        matched_uris.extend(uris)
                
                # Format results
                matched_uris = list(set(matched_uris))  # Remove duplicates
                batch_results.append(self._format_results(matched_uris, query_name))
            
            return batch_results
            
        except Exception as e:
            print(f"❌ Batch SPARQL ERROR: {e}")
            # Fallback to individual queries
            return [self.search_entity(q[0], q[1], q[2]) for q in queries]


# ============================================
# JSON Post-Processor
# ============================================

class SparnaturalPostProcessor:
    """Sparnatural JSON Post-Processing with Reconciliation"""
    
    PLACEHOLDER_URI = "https://services.sparnatural.eu/api/v1/URI_NOT_FOUND"
    
    def __init__(self, reconciliation_service: ReconciliationService):
        self.reconciliation = reconciliation_service
    
    def _collect_and_mark_placeholders(
            self, 
            obj: Any, 
            parent_type: Optional[str] = None
        ) -> Dict[str, QueryInput]:
            """
            Collect placeholders and mark their positions for later injection.
            Single-pass optimization.
            """
            placeholders = {}
            counter = [0]  
            
            def traverse(node: Any, current_parent_type: Optional[str] = None, current_parent_predicate: Optional[str] = None):
                if isinstance(node, dict) and "p" in node:
                    current_parent_predicate = node.get("p")
                if isinstance(node, dict):
                    if isinstance(node.get("values"), list):
                        line_type = node.get("oType") # or node.get("sType")

                        for criteria_item in node["values"]:
                            if isinstance(criteria_item, dict) and "rdfTerm" in criteria_item:
                                rdf_term = criteria_item["rdfTerm"]
                                if (
                                    rdf_term.get("type") == "uri"
                                    and rdf_term.get("value") == self.PLACEHOLDER_URI
                                ):
                                    label = criteria_item.get("label", "")
                                    key = f"label_{counter[0]}"
                                    counter[0] += 1
                                    
                                    placeholders[key] = QueryInput(
                                        query=label,
                                        type=line_type or current_parent_type,
                                        predicate=current_parent_predicate
                                    )
                                    
                                    # Mark position for injection (store reference)
                                    criteria_item["_placeholder_key"] = key
                    
                    for key, value in node.items():
                        traverse(value, current_parent_type, current_parent_predicate)
                
                elif isinstance(node, list):
                    for item in node:
                        traverse(item, current_parent_type, current_parent_predicate)
            
            traverse(obj, parent_type)
            return placeholders
    
    def _inject_uris_inplace(
            self, 
            obj: Any, 
            uri_mapping: Dict[str, list[str]]
        ) -> None:
            """
            Inject resolved URIs directly using marked positions.
            In-place modification, no deep copy needed.
            """
            def traverse(node: Any):
                if isinstance(node, dict):
                    if "values" in node and isinstance(node["values"], list):
                        # Collect new items to add (to avoid modifying list during iteration)
                        new_items = []
                        
                        for criteria_item in node["values"]:
                            if isinstance(criteria_item, dict):
                                # Check if this item was marked as a placeholder
                                placeholder_key = criteria_item.get("_placeholder_key")
                                
                                if placeholder_key and placeholder_key in uri_mapping:
                                    uris = uri_mapping[placeholder_key]
                                    old_label = criteria_item.get("label", "")
                                    
                                    if len(uris) >= 1:
                                        # Update first URI in place
                                        criteria_item["rdfTerm"]["value"] = uris[0]
                                        
                                        # Add additional URIs as new items
                                        for extra_uri in uris[1:]:
                                            new_items.append({
                                                "label": old_label,
                                                "rdfTerm": {
                                                    "type": "uri",
                                                    "value": extra_uri
                                                }
                                            })
                                    
                                    # Clean up marker
                                    del criteria_item["_placeholder_key"]
                        
                        # Add new items
                        if new_items:
                            node["values"].extend(new_items)

                    for value in node.values():
                        traverse(value)
                
                elif isinstance(node, list):
                    for item in node:
                        traverse(item)
            
            traverse(obj)
    
    def process(self, sparnatural_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            sparnatural_json: JSON with placeholders from LLM
        
        Returns:
            Processed JSON with URIs injected (modified in-place)
        """
        
        # Type validation
        if not isinstance(sparnatural_json, dict):
            print(f"❌ ERROR: Expected dict, got {type(sparnatural_json)}")
            print(f"Content: {sparnatural_json}")
            raise TypeError(f"sparnatural_json must be a dict, got {type(sparnatural_json).__name__}")

        # Single-pass: collect and mark placeholders
        placeholders = self._collect_and_mark_placeholders(sparnatural_json)
        
        if not placeholders:
            print("\n No placeholder found")
            return sparnatural_json
        
        # Batch reconciliation with caching
        uri_results = self.reconciliation.reconcile_batch(placeholders)
 
        # Build URI mapping
        uri_mapping = {}
        for key, results in uri_results.items():
            if results:
                uri_mapping[key] = [r.id for r in results]
            else:
                uri_mapping[key] = []

        # In-place injection (no deep copy)
        self._inject_uris_inplace(sparnatural_json, uri_mapping)
        
        return sparnatural_json
    
    def _process_branch(self, branch: Dict[str, Any], indent: int) -> List[str]:

        patterns = []
        indent_str = "  " * indent
        
        line = branch.get("line", {})
        subject = line.get("s")
        predicate = line.get("p")
        obj = line.get("o")
        s_type = line.get("sType")
        values = line.get("values", [])

        if subject and s_type:
            patterns.append(f"{indent_str}?{subject} rdf:type <{s_type}>.")

        if subject:
            patterns.append(f"{indent_str}OPTIONAL {{ ?{subject} rdfs:label ?{subject}_label. }}")

        if subject and predicate and obj:
            if values:
                for value_item in values:
                    if "rdfTerm" in value_item:
                        rdf_term = value_item["rdfTerm"]
                        if rdf_term.get("type") == "uri":
                            uri_value = rdf_term.get("value")
                            patterns.append(f"{indent_str}?{subject} <{predicate}> <{uri_value}>.")
            else:
                patterns.append(f"{indent_str}?{subject} <{predicate}> ?{obj}.")

        for child in branch.get("children", []):
            patterns.extend(self._process_branch(child, indent))
        
        return patterns

import re

def fix_label_union(sparql: str) -> str:
    """Use pre-compiled pattern for better performance"""
    def repl(match):
        subj = match.group(1)
        obj = match.group(2)
        return (
            f"{{ {subj} <http://www.w3.org/2000/01/rdf-schema#label> {obj} . }} UNION "
            f"{{ {subj} <https://schema.org/description> {obj} . }}"
        )

    return _LABEL_PATTERN.sub(repl, sparql)


class TEXT2JSON2SPARQLPipeline:
    """JSON Post-Processing and SPARQL Conversion Pipeline"""
    def __init__(self):

        self.reconciliation = ReconciliationService()
        self.post_processor = SparnaturalPostProcessor(self.reconciliation)
        self.converter = SparnaturalToSparql()
    
    def process_json_to_sparql(
        self, 
        llm_json: Dict[str, Any],
        skip_reconciliation: bool = False
    ) -> Tuple[Dict[str, Any], str]:
        
        # Type validation
        if not isinstance(llm_json, dict):
            print(f"❌ ERROR in process_json_to_sparql: Expected dict, got {type(llm_json)}")
            print(f"Content: {llm_json}")
            raise TypeError(f"llm_json must be a dict, got {type(llm_json).__name__}")

        if not skip_reconciliation:
            processed_json = self.post_processor.process(llm_json)
        else:
            processed_json = llm_json

        sparql = self.converter.convert(processed_json)
        if "Dimension" in sparql:
            sparql = fix_dimension_query(sparql)
        
        if "Label_" in sparql:
            sparql = fix_label_union(sparql)

        return processed_json, sparql
