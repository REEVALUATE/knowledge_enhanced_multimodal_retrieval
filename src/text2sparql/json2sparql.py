"""
JSON to SPARQL Converter following Sparnatural structure
"""

from typing import Dict, List, Any, Optional, Set, Union as UnionType
import json
import re
PLACEHOLDER = "https://services.sparnatural.eu/api/v1/URI_NOT_FOUND"

class SparnaturalToSparql:
    """Sparnatural JSON → SPARQL Converter"""
    
    def __init__(self):
        self.prefixes = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#"
        }
        self.query_vars: Set[str] = set()
        self.label_vars: Set[str] = set()
        self.declared_vars: Set[str] = set()  
        self.filters: List[str] = []  
        
    def convert(self, sparnatural_json: Dict[str, Any]) -> str:

        self.query_vars.clear()
        self.label_vars.clear()
        self.declared_vars.clear()  
        self.filters.clear() 

        distinct = sparnatural_json.get("distinct", True)
        variables = sparnatural_json.get("variables", [])
        branches = sparnatural_json.get("branches", [])
        order = sparnatural_json.get("order")

        for var in variables:
            if var.get("termType") == "Variable":
                var_name = var.get("value")
                if var_name:
                    self.query_vars.add(var_name)

        sparql_parts = []
        
        # 1. PREFIX  
        prefix_str = self._build_prefixes()
        if prefix_str:
            sparql_parts.append(prefix_str)
        
        # 2. SELECT
        select_clause = self._build_select(distinct)
        sparql_parts.append(select_clause)
        
        # 3. WHERE FILTER
        where_clause = self._build_where(branches)
        sparql_parts.append(where_clause)
        
        return "\n".join(sparql_parts)
    
    def _build_prefixes(self) -> str:
        """PREFIX"""
        prefix_lines = []
        for prefix, uri in sorted(self.prefixes.items()):
            prefix_lines.append(f"PREFIX {prefix}: <{uri}>")
        return "\n".join(prefix_lines)
    
    def _build_select(self, distinct: bool) -> str:
        """SELECT"""
        select_type = "SELECT DISTINCT" if distinct else "SELECT"

        vars_list = []
        for var in sorted(self.query_vars):
            vars_list.append(f"?{var}")
        
        return f"{select_type} {' '.join(vars_list)} WHERE {{"
    
    def _build_where(self, branches: List[Dict[str, Any]]) -> str:
        """WHERE"""
        patterns = []
        
        for branch in branches:
            branch_patterns = self._process_branch(branch, indent=1)
            patterns.extend(branch_patterns)
        
        # Add FILTER expressions
        if self.filters:
            for filter_expr in self.filters:
                patterns.append(f"  {filter_expr}")
        
        # WHERE closing brace
        patterns.append("}")
        
        return "\n".join(patterns)
    
    def _infer_datatype(self, value: UnionType[int, float, str]) -> str:
        """
        predict the datatype based on the value type
        """
        if isinstance(value, bool):
            return "xsd:boolean"
        elif isinstance(value, int):
            return "xsd:integer"
        elif isinstance(value, float):
            return "xsd:decimal"
        elif isinstance(value, str):
            try:
                int(value)
                return "xsd:integer"
            except ValueError:
                try:
                    float(value)
                    return "xsd:decimal"
                except ValueError:
                    if re.match(r'^\d{4}-\d{2}-\d{2}', value):
                        return "xsd:dateTime"
                    elif re.match(r'^\d{4}-\d{2}-\d{2}T', value):
                        return "xsd:dateTime"
                    else:
                        return "xsd:string"
        else:
            return "xsd:string"
    
    def _build_filter_for_range(
        self, 
        variable: str, 
        restriction: Dict[str, Any],
        indent: int = 1
    ) -> Optional[str]:
        """
        Build FILTER expression for range constraints (min/max)

        """
        min_val = restriction.get("min")
        max_val = restriction.get("max")
        
        if min_val is None and max_val is None:
            return None
        
        indent_str = "  " * indent

        datatype = self._infer_datatype(max_val if max_val is not None else min_val)
        
        conditions = []
        
        if min_val is not None:
            if datatype == "xsd:string":
                formatted_min = f'"{min_val}"'
            else:
                formatted_min = f'"{min_val}"^^{datatype}'
            conditions.append(f"?{variable} >= {formatted_min}")
        
        if max_val is not None:
            if datatype == "xsd:string":
                formatted_max = f'"{max_val}"'
            else:
                formatted_max = f'"{max_val}"^^{datatype}'
            conditions.append(f"?{variable} <= {formatted_max}")
        
        if len(conditions) == 1:
            filter_expr = f"{indent_str}FILTER({conditions[0]})"
        else:
            filter_expr = f"{indent_str}FILTER(({conditions[0]}) && ({conditions[1]}))"

        label = restriction.get("label", "")
        if label:
            filter_expr = f"{indent_str}# {label}\n{filter_expr}"
        
        return filter_expr
    
    def _process_branch(self, branch: Dict[str, Any], indent: int = 1) -> List[str]:

        patterns = []
        indent_str = "  " * indent

        is_optional = branch.get("optional", False)
        is_not_exists = branch.get("notExists", False)
        
        if "line" not in branch:
            return patterns
        
        line = branch["line"]

        subject = line.get("s")
        predicate = line.get("p")
        obj = line.get("o")
        s_type = line.get("sType")
        o_type = line.get("oType")
        values_temp = line.get("values", [])  

        values = []
        values_restriction = []
        values_literal = []
        for v in values_temp:
            if "rdfTerm" in v:
                if v["rdfTerm"].get("type") == "uri":
                    values.append(v)
                elif v["rdfTerm"].get("type") == "literal":
                    values_literal.append(v)
            else:
                values_restriction.append(v)

        if is_optional:
            patterns.append(f"{indent_str}OPTIONAL {{")
            indent += 1
            indent_str = "  " * indent
        elif is_not_exists:
            patterns.append(f"{indent_str}NOT EXISTS {{")
            indent += 1
            indent_str = "  " * indent

        if subject and s_type and subject not in self.declared_vars:
            if len(s_type) == 1:
                patterns.append(f"{indent_str}?{subject} rdf:type <{s_type[0]}>.")
                self.declared_vars.add(subject)
            elif len(s_type) > 1:  # UNION
                patterns.append(f"{indent_str}{{ ?{subject} rdf:type <{s_type[0]}>. }}")
                for st in s_type[1:]:
                    patterns.append(f"{indent_str}  UNION")
                    patterns.append(f"{indent_str}{{ ?{subject} rdf:type <{st}>. }}")
                self.declared_vars.add(subject)

        if subject and predicate and obj:
            if values:
                if len(values) > 1:

                    for i, value_item in enumerate(values):
                        if "rdfTerm" in value_item:
                            rdf_term = value_item["rdfTerm"]
                            if rdf_term.get("type") == "uri":
                                uri_value = rdf_term.get("value")
                                if uri_value == PLACEHOLDER:
                                    continue
                                label = value_item.get("label", "")

                                if i > 0:
                                    patterns.append(f"{indent_str}  UNION")
                                
                                patterns.append(f"{indent_str}  {{ ?{subject} <{predicate}> <{uri_value}>. }} # {label}")

                else:
                    for value_item in values:
                        if "rdfTerm" in value_item:
                            rdf_term = value_item["rdfTerm"]
                            if rdf_term.get("type") == "uri":
                                uri_value = rdf_term.get("value")
                                if uri_value == PLACEHOLDER:
                                    continue
                                label = value_item.get("label", "")
                                patterns.append(f"{indent_str}# {label}")
                                patterns.append(f"{indent_str}?{subject} <{predicate}> <{uri_value}>.")
            else:
                patterns.append(f"{indent_str}?{subject} <{predicate}> ?{obj}.")

                if o_type and obj not in self.declared_vars:
                    if len(o_type) == 1:
                        patterns.append(f"{indent_str}?{obj} rdf:type <{o_type[0]}>.")
                        self.declared_vars.add(obj)
                    elif len(o_type) > 1:  # UNION
                        
                        patterns.append(f"{indent_str}{{ ?{obj} rdf:type <{o_type[0]}>. }}")
                        for ot in o_type[1:]:
                            patterns.append(f"{indent_str}  UNION")
                            patterns.append(f"{indent_str}{{ ?{obj} rdf:type <{ot}>. }}")
                        self.declared_vars.add(obj)
            
            if values_literal and obj:
                for i, value_item in enumerate(values_literal):
                    if "rdfTerm" in value_item:
                        rdf_term = value_item["rdfTerm"]
                        if rdf_term.get("type") == "literal":
                            datatype = self._infer_datatype(rdf_term.get("value"))
                            lit_value = rdf_term.get("value")

                            if datatype:
                                if datatype == "xsd:string":
                                    formatted_value = f'"{lit_value}"@en'
                                else:
                                    formatted_value = f'"{lit_value}"^^{datatype}'
                            else:
                                formatted_value = f'"{lit_value}"'
                            # add FILTER to let the number equal
                            patterns.append(f"{indent_str}  FILTER(?{obj} = {formatted_value})")
        
            if values_restriction and obj:
                for restriction in values_restriction:
                    filter_expr = self._build_filter_for_range(obj, restriction, indent)
                    if filter_expr:
                        self.filters.append(filter_expr)

        if "children" in branch and branch["children"]:
            for child in branch["children"]:
                child_patterns = self._process_branch(child, indent)
                patterns.extend(child_patterns)

        if is_optional or is_not_exists:
            indent -= 1
            indent_str = "  " * indent
            patterns.append(f"{indent_str}}}")
        
        return patterns
    

    def add_prefix(self, prefix: str, uri: str):

        self.prefixes[prefix] = uri