# DSPy Signatures/Modules

# agent/dspy_signatures.py

from typing import Dict, Any, List
import random
import subprocess
import shutil
import json
import shlex
import re
from collections import defaultdict, Counter
import dspy

# ------------------------------
# Router Module (DSPy)
# ------------------------------
class RouterModule:
    """
    Decide which path to take: 'rag', 'sql', or 'hybrid'.
    DSPy can optimize this classifier.
    """

    def __init__(self):
        # Example: simple keyword-based; replace with DSPy classifier later
        self.routes = ["rag", "sql", "hybrid"]

    def __call__(self, question: str) -> str:
        q = question.lower()
        if any(k in q for k in ["return", "policy", "marketing", "catalog"]):
            return "rag"
        elif any(k in q for k in ["top", "sum", "average", "aov", "revenue"]):
            return "sql"
        else:
            return "hybrid"


# ------------------------------
# NL→SQL Module (DSPy + phi3)
# ------------------------------
class NL2SQLModule:
    """
    Generate SQLite queries from natural language + optional constraints.
    Uses template fallback; prefers local Ollama phi3 if available for better SQL.
    """

    def __init__(self, use_ollama: bool = True):
        # Example: small template-based SQL
        self.templates = {
            "aov": """
                SELECT AVG(OrderRevenue) AS AOV
                FROM (
                    SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS OrderRevenue
                    FROM Orders o
                    JOIN "Order Details" od ON o.OrderID = od.OrderID
                    {where_clause}
                    GROUP BY o.OrderID
                )
            """,
            "top_products": """
                SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue
                FROM "Order Details" od
                JOIN Products p ON od.ProductID = p.ProductID
                {where_clause}
                GROUP BY p.ProductName
                ORDER BY Revenue DESC
                LIMIT 5
            """
        }
        self.use_ollama = use_ollama
        self.ollama = OllamaClient() if use_ollama else None
        # keep Ollama client only if it has either a dspy model or CLI fallback
        if self.ollama and not (getattr(self.ollama, "dspy_model", None) or getattr(self.ollama, "cli_available", False)):
            self.ollama = None  # fallback silently

    def _extract_sql_from_text(self, text: str) -> str:
        """Heuristic: return substring starting at first SELECT."""
        if not text:
            return ""
        idx = text.lower().find("select")
        if idx >= 0:
            return text[idx:].strip()
        return ""

    def __call__(self, question: str, constraints: Dict[str, Any]) -> str:
        q = question.lower()
        where_clause = ""
        if "year" in constraints:
            where_clause = f"WHERE strftime('%Y', o.OrderDate) = '{constraints['year']}'"

        # Prefer Ollama-generated SQL when available
        if self.ollama:
            prompt = (
                "You are an expert at writing valid SQLite queries against the Northwind schema. "
                "Generate a single valid SQLite query (no explanation) that answers the question below. "
                "Use tables Orders, \"Order Details\", Products when relevant. "
                f"Question: {question}\n"
                f"Constraints: {json.dumps(constraints)}\n"
                "Return only the SQL statement."
            )
            out = self.ollama.generate(prompt)
            sql_candidate = self._extract_sql_from_text(out)
            # Basic validation: ensure it contains SELECT
            if sql_candidate and "select" in sql_candidate.lower():
                return sql_candidate
            # otherwise fall through to templates

        # templated fallback
        if "aov" in q or "average order value" in q:
            return self.templates["aov"].format(where_clause=where_clause)
        elif "top products" in q or "revenue" in q or "top 3 products" in q:
            return self.templates["top_products"].format(where_clause=where_clause)
        else:
            # fallback simple query
            return "SELECT 1;"


# ------------------------------
# Synthesizer Module (DSPy + phi3)
# ------------------------------
class SynthesizerModule:
    """
    Produce typed answer with citations and confidence.
    Prefers local Ollama phi3 when available to format outputs exactly as required
    (e.g., matching format_hint and producing explicit citations).
    """

    def __init__(self, use_ollama: bool = True):
        self.use_ollama = use_ollama
        self.ollama = OllamaClient() if use_ollama else None
        # keep Ollama client only if it has either a dspy model or CLI fallback
        if self.ollama and not (getattr(self.ollama, "dspy_model", None) or getattr(self.ollama, "cli_available", False)):
            self.ollama = None

    def _short_docs_snippet(self, docs: List[Dict[str, Any]], max_chars: int = 800) -> str:
        snippets = []
        for d in docs[:3]:
            if isinstance(d, dict):
                id_ = d.get("chunk_id") or d.get("id") or ""
                txt = d.get("content") or d.get("text") or ""
                snippets.append(f"[{id_}] {txt[:300]}")
            else:
                snippets.append(str(d)[:300])
        return "\n\n".join(snippets)[:max_chars]

    def __call__(self, question: str, sql_result: Dict[str, Any], docs: List[Dict[str, Any]]):
        # If Ollama available, prefer it to synthesize a structured JSON answer.
        if self.ollama:
            # Build a compact context and ask phi3 to return JSON with final_answer, citations, confidence.
            context = {
                "question": question,
                "sql_result": {
                    "columns": sql_result.get("columns"),
                    "rows": sql_result.get("rows"),
                    "success": sql_result.get("success", False),
                    "error": sql_result.get("error")
                },
                "docs_snippet": self._short_docs_snippet(docs)
            }
            prompt = (
                "You are an assistant producing typed, auditable answers for a retail analytics system. "
                "Given the context, produce a single JSON object with keys: final_answer, citations (list), confidence (0.0-1.0). "
                "final_answer must strictly match the requested type implied in the question (e.g., 'Return an integer', 'Return a float', 'Return {customer:str, margin:float}', 'Return list[{product:str, revenue:float}]'). "
                "Citations must include every DB table used and every doc chunk id you relied on (format: source::chunk_id). "
                "Do not include any extra fields. Return only valid JSON.\n\n"
                f"Context: {json.dumps(context)}\n\nQuestion: {question}\n\n"
                "Output JSON:"
            )
            out = self.ollama.generate(prompt)
            # Try to extract JSON from the model output
            # Heuristic: find first '{' and last '}' and parse
            try:
                start = out.find("{")
                end = out.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_text = out[start:end+1]
                    parsed = json.loads(json_text)
                    fa = parsed.get("final_answer")
                    citations = parsed.get("citations", [])
                    confidence = float(parsed.get("confidence", 0.0))
                    return fa, citations, confidence
            except Exception:
                # fall through to classic behavior
                pass

        # Fallback (existing simple behavior)
        if sql_result.get("success") and sql_result.get("rows"):
            rows = sql_result["rows"]
            # pick first row if single value
            if len(rows) == 1 and len(rows[0]) == 1:
                try:
                    final_answer = float(rows[0][0])
                except Exception:
                    final_answer = rows[0][0]
            else:
                final_answer = rows  # return list of rows
        else:
            # fallback to doc content
            if docs:
                first = docs[0]
                if isinstance(first, dict):
                    content = first.get("content") or first.get("text") or str(first)
                else:
                    content = str(first)
                final_answer = content
            else:
                final_answer = "No answer found."

        citations = []
        if sql_result.get("success"):
            citations.append("Orders")
            citations.append("Order Details")
            if "ProductName" in sql_result.get("columns", []):
                citations.append("Products")
        # Add doc chunk ids if present; otherwise add a lightweight doc reference
        for d in (docs or []):
            if isinstance(d, dict):
                if "chunk_id" in d:
                    citations.append(d["chunk_id"])
                elif "id" in d:
                    citations.append(d["id"])
                else:
                    excerpt = None
                    if "content" in d:
                        excerpt = (d["content"][:50] + "...") if len(d["content"])>50 else d["content"]
                    elif "text" in d:
                        excerpt = (d["text"][:50] + "...") if len(d["text"])>50 else d["text"]
                    else:
                        excerpt = str(d)[:50]
                    citations.append(f"doc:{excerpt}")
            else:
                citations.append(str(d)[:50])

        confidence = 0.9 if sql_result.get("success") else (0.75 if docs else 0.6)

        return final_answer, citations, confidence

# Add dspy-based Signatures & Predict wrappers when available
if _dspy_available:
    try:
        # Lightweight signatures matching module inputs/outputs
        class RouterSignature(dspy.Signature):
            question: str = dspy.InputField()
            label: str = dspy.OutputField()

        class NL2SQLSignature(dspy.Signature):
            question: str = dspy.InputField()
            constraints: dict = dspy.InputField()
            sql: str = dspy.OutputField()

        class SynthesizerSignature(dspy.Signature):
            question: str = dspy.InputField()
            sql_result: dict = dspy.InputField()
            docs_snippet: str = dspy.InputField()
            format_hint: str = dspy.InputField()
            final_answer: Any = dspy.OutputField()
            citations: list = dspy.OutputField()
            confidence: float = dspy.OutputField()

        # Instantiate Predict callables (will use dspy.settings configured LM)
        RouterPredict = dspy.Predict(RouterSignature)
        NL2SQLPredict = dspy.Predict(NL2SQLSignature)
        SynthPredict = dspy.Predict(SynthesizerSignature)

        # Override modules to use dspy Predict, with robust fallbacks
        class RouterModule:
            def __init__(self):
                # create Predict instance (uses configured LM)
                self._pred = RouterPredict()

            def __call__(self, question: str) -> str:
                try:
                    resp = self._pred(question=question)
                    # dspy response may expose attributes or dict-like
                    label = getattr(resp, "label", None) if resp is not None else None
                    if label is None and isinstance(resp, dict):
                        label = resp.get("label")
                    if label:
                        return label
                except Exception:
                    pass
                # fallback heuristic
                q = question.lower()
                if any(k in q for k in ["return", "policy", "marketing", "catalog"]):
                    return "rag"
                elif any(k in q for k in ["top", "sum", "average", "aov", "revenue"]):
                    return "sql"
                return "hybrid"

        class NL2SQLModule:
            def __init__(self, use_ollama: bool = True):
                # keep existing templates
                self.templates = {
                    "aov": """
                        SELECT AVG(OrderRevenue) AS AOV
                        FROM (
                            SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS OrderRevenue
                            FROM Orders o
                            JOIN "Order Details" od ON o.OrderID = od.OrderID
                            {where_clause}
                            GROUP BY o.OrderID
                        )
                    """,
                    "top_products": """
                        SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue
                        FROM "Order Details" od
                        JOIN Products p ON od.ProductID = p.ProductID
                        {where_clause}
                        GROUP BY p.ProductName
                        ORDER BY Revenue DESC
                        LIMIT 5
                    """
                }
                self._pred = NL2SQLPredict()

            def _extract_sql_from_text(self, text: str) -> str:
                if not text:
                    return ""
                idx = text.lower().find("select")
                if idx >= 0:
                    return text[idx:].strip()
                return ""

            def __call__(self, question: str, constraints: Dict[str, Any]) -> str:
                # try dspy-predict first
                try:
                    resp = self._pred(question=question, constraints=constraints)
                    sql = getattr(resp, "sql", None)
                    if sql is None and isinstance(resp, dict):
                        sql = resp.get("sql")
                    if sql:
                        return self._extract_sql_from_text(sql) or sql
                except Exception:
                    pass
                # fallback: template logic
                q = question.lower()
                where_clause = ""
                if "year" in constraints:
                    where_clause = f"WHERE strftime('%Y', o.OrderDate) = '{constraints['year']}'"
                if "aov" in q or "average order value" in q:
                    return self.templates["aov"].format(where_clause=where_clause)
                elif "top products" in q or "revenue" in q or "top 3 products" in q:
                    return self.templates["top_products"].format(where_clause=where_clause)
                return "SELECT 1;"

        class SynthesizerModule:
            def __init__(self):
                self._pred = SynthPredict()

            def _short_docs_snippet(self, docs: List[Dict[str, Any]], max_chars: int = 800) -> str:
                snippets = []
                for d in docs[:3]:
                    if isinstance(d, dict):
                        id_ = d.get("chunk_id") or d.get("id") or ""
                        txt = d.get("content") or d.get("text") or ""
                        snippets.append(f"[{id_}] {txt[:300]}")
                    else:
                        snippets.append(str(d)[:300])
                return "\n\n".join(snippets)[:max_chars]

            def __call__(self, question: str, sql_result: Dict[str, Any], docs: List[Dict[str, Any]]):
                try:
                    snippet = self._short_docs_snippet(docs)
                    fmt_hint = ""  # GraphHybrid forwards format_hint into Synthesizer via state if you decide; keep blank safe default
                    resp = self._pred(question=question, sql_result=sql_result, docs_snippet=snippet, format_hint=fmt_hint)
                    # Extract outputs from dspy response
                    fa = getattr(resp, "final_answer", None)
                    cits = getattr(resp, "citations", None)
                    conf = getattr(resp, "confidence", None)
                    if fa is None and isinstance(resp, dict):
                        fa = resp.get("final_answer")
                        cits = resp.get("citations", [])
                        conf = resp.get("confidence", 0.0)
                    if fa is not None:
                        return fa, (cits or []), float(conf or 0.0)
                except Exception:
                    pass
                # fallback to previous simple behavior
                if sql_result.get("success") and sql_result.get("rows"):
                    rows = sql_result["rows"]
                    if len(rows) == 1 and len(rows[0]) == 1:
                        try:
                            final_answer = float(rows[0][0])
                        except Exception:
                            final_answer = rows[0][0]
                    else:
                        final_answer = rows
                else:
                    if docs:
                        first = docs[0]
                        if isinstance(first, dict):
                            content = first.get("content") or first.get("text") or str(first)
                        else:
                            content = str(first)
                        final_answer = content
                    else:
                        final_answer = "No answer found."
                citations = []
                if sql_result.get("success"):
                    citations.append("Orders")
                    citations.append("Order Details")
                    if "ProductName" in sql_result.get("columns", []):
                        citations.append("Products")
                for d in (docs or []):
                    if isinstance(d, dict):
                        if "chunk_id" in d:
                            citations.append(d["chunk_id"])
                        elif "id" in d:
                            citations.append(d["id"])
                        else:
                            excerpt = None
                            if "content" in d:
                                excerpt = (d["content"][:50] + "...") if len(d["content"])>50 else d["content"]
                            elif "text" in d:
                                excerpt = (d["text"][:50] + "...") if len(d["text"])>50 else d["text"]
                            else:
                                excerpt = str(d)[:50]
                            citations.append(f"doc:{excerpt}")
                    else:
                        citations.append(str(d)[:50])
                confidence = 0.9 if sql_result.get("success") else (0.75 if docs else 0.6)
                return final_answer, citations, confidence

    except Exception:
        # If any dspy wiring fails, fall back to classic class definitions below
        pass

# ------------------------------
# Router Module (DSPy)
# ------------------------------
class RouterModule:
    """
    Decide which path to take: 'rag', 'sql', or 'hybrid'.
    DSPy can optimize this classifier.
    """

    def __init__(self):
        # Example: simple keyword-based; replace with DSPy classifier later
        self.routes = ["rag", "sql", "hybrid"]

    def __call__(self, question: str) -> str:
        q = question.lower()
        if any(k in q for k in ["return", "policy", "marketing", "catalog"]):
            return "rag"
        elif any(k in q for k in ["top", "sum", "average", "aov", "revenue"]):
            return "sql"
        else:
            return "hybrid"


# ------------------------------
# NL→SQL Module (DSPy + phi3)
# ------------------------------
class NL2SQLModule:
    """
    Generate SQLite queries from natural language + optional constraints.
    Uses template fallback; prefers local Ollama phi3 if available for better SQL.
    """

    def __init__(self, use_ollama: bool = True):
        # Example: small template-based SQL
        self.templates = {
            "aov": """
                SELECT AVG(OrderRevenue) AS AOV
                FROM (
                    SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS OrderRevenue
                    FROM Orders o
                    JOIN "Order Details" od ON o.OrderID = od.OrderID
                    {where_clause}
                    GROUP BY o.OrderID
                )
            """,
            "top_products": """
                SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue
                FROM "Order Details" od
                JOIN Products p ON od.ProductID = p.ProductID
                {where_clause}
                GROUP BY p.ProductName
                ORDER BY Revenue DESC
                LIMIT 5
            """
        }
        self.use_ollama = use_ollama
        self.ollama = OllamaClient() if use_ollama else None
        # keep Ollama client only if it has either a dspy model or CLI fallback
        if self.ollama and not (getattr(self.ollama, "dspy_model", None) or getattr(self.ollama, "cli_available", False)):
            self.ollama = None  # fallback silently

    def _extract_sql_from_text(self, text: str) -> str:
        """Heuristic: return substring starting at first SELECT."""
        if not text:
            return ""
        idx = text.lower().find("select")
        if idx >= 0:
            return text[idx:].strip()
        return ""

    def __call__(self, question: str, constraints: Dict[str, Any]) -> str:
        q = question.lower()
        where_clause = ""
        if "year" in constraints:
            where_clause = f"WHERE strftime('%Y', o.OrderDate) = '{constraints['year']}'"

        # Prefer Ollama-generated SQL when available
        if self.ollama:
            prompt = (
                "You are an expert at writing valid SQLite queries against the Northwind schema. "
                "Generate a single valid SQLite query (no explanation) that answers the question below. "
                "Use tables Orders, \"Order Details\", Products when relevant. "
                f"Question: {question}\n"
                f"Constraints: {json.dumps(constraints)}\n"
                "Return only the SQL statement."
            )
            out = self.ollama.generate(prompt)
            sql_candidate = self._extract_sql_from_text(out)
            # Basic validation: ensure it contains SELECT
            if sql_candidate and "select" in sql_candidate.lower():
                return sql_candidate
            # otherwise fall through to templates

        # templated fallback
        if "aov" in q or "average order value" in q:
            return self.templates["aov"].format(where_clause=where_clause)
        elif "top products" in q or "revenue" in q or "top 3 products" in q:
            return self.templates["top_products"].format(where_clause=where_clause)
        else:
            # fallback simple query
            return "SELECT 1;"


# ------------------------------
# Synthesizer Module (DSPy + phi3)
# ------------------------------
class SynthesizerModule:
    """
    Produce typed answer with citations and confidence.
    Prefers local Ollama phi3 when available to format outputs exactly as required
    (e.g., matching format_hint and producing explicit citations).
    """

    def __init__(self, use_ollama: bool = True):
        self.use_ollama = use_ollama
        self.ollama = OllamaClient() if use_ollama else None
        # keep Ollama client only if it has either a dspy model or CLI fallback
        if self.ollama and not (getattr(self.ollama, "dspy_model", None) or getattr(self.ollama, "cli_available", False)):
            self.ollama = None

    def _short_docs_snippet(self, docs: List[Dict[str, Any]], max_chars: int = 800) -> str:
        snippets = []
        for d in docs[:3]:
            if isinstance(d, dict):
                id_ = d.get("chunk_id") or d.get("id") or ""
                txt = d.get("content") or d.get("text") or ""
                snippets.append(f"[{id_}] {txt[:300]}")
            else:
                snippets.append(str(d)[:300])
        return "\n\n".join(snippets)[:max_chars]

    def __call__(self, question: str, sql_result: Dict[str, Any], docs: List[Dict[str, Any]]):
        # If Ollama available, prefer it to synthesize a structured JSON answer.
        if self.ollama:
            # Build a compact context and ask phi3 to return JSON with final_answer, citations, confidence.
            context = {
                "question": question,
                "sql_result": {
                    "columns": sql_result.get("columns"),
                    "rows": sql_result.get("rows"),
                    "success": sql_result.get("success", False),
                    "error": sql_result.get("error")
                },
                "docs_snippet": self._short_docs_snippet(docs)
            }
            prompt = (
                "You are an assistant producing typed, auditable answers for a retail analytics system. "
                "Given the context, produce a single JSON object with keys: final_answer, citations (list), confidence (0.0-1.0). "
                "final_answer must strictly match the requested type implied in the question (e.g., 'Return an integer', 'Return a float', 'Return {customer:str, margin:float}', 'Return list[{product:str, revenue:float}]'). "
                "Citations must include every DB table used and every doc chunk id you relied on (format: source::chunk_id). "
                "Do not include any extra fields. Return only valid JSON.\n\n"
                f"Context: {json.dumps(context)}\n\nQuestion: {question}\n\n"
                "Output JSON:"
            )
            out = self.ollama.generate(prompt)
            # Try to extract JSON from the model output
            # Heuristic: find first '{' and last '}' and parse
            try:
                start = out.find("{")
                end = out.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_text = out[start:end+1]
                    parsed = json.loads(json_text)
                    fa = parsed.get("final_answer")
                    citations = parsed.get("citations", [])
                    confidence = float(parsed.get("confidence", 0.0))
                    return fa, citations, confidence
            except Exception:
                # fall through to classic behavior
                pass

        # Fallback (existing simple behavior)
        if sql_result.get("success") and sql_result.get("rows"):
            rows = sql_result["rows"]
            # pick first row if single value
            if len(rows) == 1 and len(rows[0]) == 1:
                try:
                    final_answer = float(rows[0][0])
                except Exception:
                    final_answer = rows[0][0]
            else:
                final_answer = rows  # return list of rows
        else:
            # fallback to doc content
            if docs:
                first = docs[0]
                if isinstance(first, dict):
                    content = first.get("content") or first.get("text") or str(first)
                else:
                    content = str(first)
                final_answer = content
            else:
                final_answer = "No answer found."

        citations = []
        if sql_result.get("success"):
            citations.append("Orders")
            citations.append("Order Details")
            if "ProductName" in sql_result.get("columns", []):
                citations.append("Products")
        # Add doc chunk ids if present; otherwise add a lightweight doc reference
        for d in (docs or []):
            if isinstance(d, dict):
                if "chunk_id" in d:
                    citations.append(d["chunk_id"])
                elif "id" in d:
                    citations.append(d["id"])
                else:
                    excerpt = None
                    if "content" in d:
                        excerpt = (d["content"][:50] + "...") if len(d["content"])>50 else d["content"]
                    elif "text" in d:
                        excerpt = (d["text"][:50] + "...") if len(d["text"])>50 else d["text"]
                    else:
                        excerpt = str(d)[:50]
                    citations.append(f"doc:{excerpt}")
            else:
                citations.append(str(d)[:50])

        confidence = 0.9 if sql_result.get("success") else (0.75 if docs else 0.6)

        return final_answer, citations, confidence
