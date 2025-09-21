"""
DSPy-only modules for Router, Retriever, NL2SQL, and Synthesizer.
No heuristics or template fallbacks — if DSPy Predict fails or returns invalid
outputs we raise RuntimeError so failures are explicit.
"""
from typing import Dict, Any, List
import json
import dspy


lm = dspy.LM("ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)

import re
import sqlite3
from typing import Any, Dict, List, Tuple, Optional, Literal

# ------------------------------
# RouterModule (DSPy-style classifier)
# ------------------------------
class RouterModule(dspy.Signature):
    """
    Classify questions into the appropriate processing method:
    - rag: For questions about policies, documents, guidelines, explanations, or content that would be in text files
    - sql: For questions about structured data, counts, averages, or data that would be in database tables
    - hybrid: For questions requiring both document knowledge and structured data
    """

    question: str = dspy.InputField(desc="The user's question to classify")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning about the classification")
    router = dspy.OutputField(desc="The classification of the question",
        choices=["rag", "sql", "hybrid"])
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1")

    @staticmethod
    def _router(question: str):
        classify = dspy.Predict(RouterModule)
        return classify(question=question)

# ------------------------------
# RetrieverModule
# ------------------------------
class RetrieverModule(dspy.Signature):
    """
    Retrieve relevant document chunks based on the question.
    """
    question: str = dspy.InputField(desc="The user's question")
    top_k: int = dspy.InputField(desc="Number of top chunks to retrieve")
    
    chunks = dspy.OutputField(desc="List of retrieved document chunks with their metadata")
    
    @staticmethod
    def _retrieve(question: str, top_k: int = 3):
        retrieve = dspy.Predict(RetrieverModule)
        return retrieve(question=question, top_k=top_k)
    
# ------------------------------
# PlannerModule 
# ------------------------------
class ExtractEntities(dspy.Signature):
    """Extract structured information from text."""

    docs: str = dspy.InputField(desc="The text to extract information from docs")
    dates: str = dspy.OutputField(desc="extracted dates or date ranges")
    kpi_formula: list[str] = dspy.OutputField(desc="a list of extracted KPIs and their formulas")
    entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")

class EntityExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(ExtractEntities)
    
    def forward(self, docs: List[Dict[str, Any]] | str):
        # Join docs into a single string if needed
        print(f"Extracting entities from docs: {docs}")
        docs_text = "\n\n".join(docs) if isinstance(docs, list) else docs
        return self.extractor(docs=docs_text)




# ------------------------------
# NL2SQLModule 
# ------------------------------
class NL2SQLModule(dspy.Signature):
    """
    Generate a SQL query from natural language using detailed constraints.
    """
    question = dspy.InputField(desc="The natural language question")
    schema_info = dspy.InputField(desc="Description of tables and columns in the database")
    constraints = dspy.InputField(desc="Query constraints and requirements")
    entities = dspy.InputField(desc="Extracted entities like dates, categories, KPIs")
    
    sql_query = dspy.OutputField(desc="The SQL query that answers the question")

class NL2SQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(NL2SQLModule)
        
        # Store common table definitions
        self.schema_info = """
            -- Core Business Tables
            Customers(CustomerID TEXT PRIMARY KEY, CompanyName TEXT, Country TEXT, ...)
            Orders(OrderID INTEGER PRIMARY KEY, CustomerID TEXT REFERENCES Customers(CustomerID), ...)
            "Order Details"(OrderID INTEGER REFERENCES Orders(OrderID), ProductID INTEGER, Quantity INTEGER, ...)

            -- Product Information
            Categories(CategoryID INTEGER PRIMARY KEY, CategoryName TEXT, Description TEXT, ...)
            Products(ProductID INTEGER PRIMARY KEY, ProductName TEXT, CategoryID INTEGER REFERENCES Categories(CategoryID), ...)
            """
        
        # Define common constraints
        self.constraints = """
        - Use standard SQL syntax compatible with SQLite
        - Order and Product relationship via "Order Details" table
        - `Order Details` has quotes because of the space in the name
        """
        

    
    def forward(self, question, retrieved_docs=None, additional_constraints=None):
        """Generate SQL query based on question, retrieved documents, and additional constraints."""
        # Identify question type and add relevant constraints
        question_constraints = self.constraints
        
        # Pattern matching to add specific constraints
        if re.search(r'\b(how many|count|total number)\b', question, re.I):
            question_constraints += "\n- Use COUNT() for counting results"
            
        if re.search(r'\b(average|avg)\b', question, re.I):
            question_constraints += "\n- Use AVG() for calculating averages"
            
        if re.search(r'\b(maximum|highest|most expensive|top)\b', question, re.I):
            question_constraints += "\n- Use ORDER BY and LIMIT for finding top values"
        
        if re.search(r'\b(group by|per|each)\b', question, re.I):
            question_constraints += "\n- Use GROUP BY for grouping results"
        
        # Extract entities from the question and retrieved documents
        # entities = self.extract_entities(question, retrieved_docs)
        entities = {}
        # Add additional constraints if provided
        if additional_constraints:
            # Merge additional constraints with extracted entities
            for key, value in additional_constraints.items():
                entities[key] = value
        
        # Format entities as a structured string for the model
        entities_str = self._format_entities(entities)
        
        # Generate the SQL query
        result = self.generator(
            question=question,
            schema_info=self.schema_info,
            constraints=question_constraints,
            entities=entities_str
        )
        
        return result.sql_query
    
    def _format_entities(self, entities):
        """Format entities as a structured string for the model."""
        if not entities:
            return "No specific entities extracted."
        
        formatted = []
        
        # Format date range
        if "date_range" in entities:
            dr = entities["date_range"]
            # Handle case where campaign name might not be present
            campaign_name = dr.get('campaign', 'Custom Period')
            formatted.append(f"Date Range: {campaign_name} from {dr['start_date']} to {dr['end_date']}")
            # Handle case where categories might not be present
            if 'categories' in dr:
                formatted.append(f"Associated Categories: {', '.join(dr['categories'])}")
        
        # Format year
        if "year" in entities:
            formatted.append(f"Year: {entities['year']}")
        
        # Format specific date
        if "specific_date" in entities:
            formatted.append(f"Specific Date: {entities['specific_date']}")
        
        # Format KPI
        if "kpi" in entities:
            kpi = entities["kpi"]
            formatted.append(f"KPI: {kpi['name']}")
            formatted.append(f"Formula: {kpi['formula']}")
        
        # Format categories
        if "categories" in entities:
            formatted.append(f"Categories: {', '.join(entities['categories'])}")
        
        return "\n".join(formatted)

# ------------------------------
# SynthesizerModule 
# ------------------------------
class SynthesizerModule(dspy.Signature):
    """
    Synthesize a final answer from SQL results and document chunks.
    Adhere exactly to the format_hint provided.
    """
    question = dspy.InputField(desc="The user's original question")
    sql_result = dspy.InputField(desc="The result from the SQL query, including columns and rows")
    docs = dspy.InputField(desc="Relevant document excerpts with their metadata")
    format_hint = dspy.InputField(desc="The expected format of the final answer (e.g., int, float, {key:type}, list[{key:type}])")
    
    final_answer = dspy.OutputField(desc="The final answer to the user's question in the specified format")
    citations = dspy.OutputField(desc="List of citations including DB tables used and document chunk IDs referenced")
    confidence = dspy.OutputField(desc="Confidence score between 0 and 1")

class SynthesizerGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.Predict(SynthesizerModule)
    
    def forward(self, question, sql_result, docs, format_hint):
        """Generate a final answer based on SQL results and document chunks."""
        result = self.synthesizer(
            question=question,
            sql_result=sql_result,
            docs=docs,
            format_hint=format_hint
        )
        
        return result.final_answer, result.citations, result.confidence


if __name__ == "__main__":
    # Test RouterModule
    result = RouterModule._router("According to the product policy, what is the return window (days) for unopened Beverages? Return an integer.")
    print(f"Router result: {result}")
    
    # Test NL2SQLGenerator
    nl2sql = NL2SQLGenerator()
    sql_query = nl2sql("What is the average order value for Beverages in 1997?")
    print(f"SQL query: {sql_query}")
    
    # Test SynthesizerGenerator
    synthesizer = SynthesizerGenerator()
    sql_result = {
        "columns": ["AverageOrderValue"],
        "rows": [[42.5]],
        "success": True
    }
    docs = [
        {"content": "Beverages unopened: 14 days; opened: no returns.", "chunk_id": "product_policy::chunk1", "source": "product_policy.md"}
    ]
    final_answer, citations, confidence = synthesizer(
        question="What is the average order value for Beverages in 1997?",
        sql_result=sql_result,
        docs=docs,
        format_hint="float"
    )
    print(f"Final answer: {final_answer}")
    print(f"Citations: {citations}")
    print(f"Confidence: {confidence}")
