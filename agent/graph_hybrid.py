# LangGraph (≥6 nodes + repair loop)

import sqlite3
import operator

from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, Literal, Annotated
from agent.rag.retrieval import DocumentRetriever
from agent.dspy_signatures import RouterModule, NL2SQLModule, SynthesizerModule, RetrieverModule, NL2SQLGenerator, SynthesizerGenerator,EntityExtractor
from tools.sqlite_tool import SQLiteTool
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod



class GraphHybrid:
    # ------------------------------
    # State object (mutable dict)
    # ------------------------------
    class AgentState(dict):
        """Defines the state that flows through the graph."""
        question: str
        route: Literal['rag', 'sql', 'hybrid', 'end'] = "end"
        docs: list[{"chunk_id","content","source", "score"}]
        constraints: dict
        sql: Annotated[str, lambda _, new_value: new_value]
        sql_result: dict
        final_answer: Annotated[Any, lambda _, new_value: new_value]
        citations: Annotated[list, lambda _, new_value: new_value]
        errors: list
        repair_attempts: Annotated[int, lambda _, new_value: new_value]
        confidence: Annotated[float, lambda _, new_value: new_value]

    def __init__(self):
        # Initialize tools
        self.retriever = DocumentRetriever()
        # RouterModule is a DSPy Signature, not a class, so we don't instantiate it directly
        self.router = RouterModule._router  # Use the static method
        self.sql_generator = NL2SQLGenerator()
        self.synthesizer = SynthesizerGenerator()
        self.graph = self._build_graph()


    # ------------------------------
    # Graph Nodes
    # ------------------------------

    def _node_router(self,state: AgentState):
        # use injected router instance; mutate state in-place and return only changed keys
        result = self.router(state["question"])
        # Extract the router value from the result
        if hasattr(result, 'router'):
            decision = result.router
        else:
            # Default to 'rag' if we can't extract the router value
            decision = 'rag'
        # state["route"] = decision
        state.setdefault("trace", []).append(f"Router → {decision}")
        print(f"Routing decision: {result}")
        return {"route": decision, "trace": state["trace"]}


    def _node_retriever(self,state: AgentState):
        # Retriever node returns list[dict] with {chunk_id, content, source, score}
        docs = self.retriever(state["question"], top_k=3)
        # normalize key names so downstream modules find docs reliably
        # state["retrieved_docs"] = docs
        # state["docs"] = docs
        state.setdefault("trace", []).append(f"Retriever → {len(docs)} chunks")
        print(f"Retrieved {len(docs)} documents. and its content is {docs}")
        return {"retrieved_docs": docs, "docs": docs, "trace": state["trace"]}


    def _node_planner(self,state: AgentState):
        # If route is 'rag', no need to plan for SQL
        route = state.get('route')
        if route == 'rag' or (hasattr(route, 'router') and route.router == 'rag'):
            return {"trace": state.get("trace", [])}
            
        # Simple constraint extraction (dates, categories, KPIs)
        # In a real case, use regex/NLP
        q = state["question"].lower()
        constraints = {}

        # Extract entities from the question and retrieved documents using the NL2SQLGenerator
        docs = state.get("docs", state.get("retrieved_docs", []))
        content_list = [content['content'] for content in docs]
        print(f"Content list for entity extraction: {content_list}")
        # entities = self.sql_generator.extract_entities(q, docs)
        extractor = EntityExtractor()
        entities = extractor(content_list)
        print(f"Extracted entities are {entities}")
        
        # Merge extracted entities with constraints
        for key, value in entities.items():
            constraints[key] = value
            
        # state["constraints"] = constraints
        state.setdefault("trace", []).append(f"Planner → {constraints}")
        return {"constraints": constraints, "trace": state["trace"]}


    def _node_nl2sql(self,state: AgentState):
        # If route is 'rag', no need to generate SQL
        route = state.get('route')
        if route == 'rag' or (hasattr(route, 'router') and route.router == 'rag'):
            return {"trace": state.get("trace", [])}
            
        # Get retrieved documents if available
        docs = state.get("docs", state.get("retrieved_docs", []))
        # Generate SQL using the enhanced NL2SQLGenerator
        sql = self.sql_generator(
            question=state["question"], 
            retrieved_docs=docs, 
            additional_constraints=state.get("constraints", {})
        )
        # state["sql"] = sql
        state.setdefault("trace", []).append(f"NL→SQL → {sql}")
        print(f"Generated SQL: {sql}")
        return {"sql": sql, "trace": state["trace"], "repair_attempts": state.get("repair_attempts", 0)}


    def _node_executor(self,state: AgentState):
        # If route is 'rag', no need to execute SQL
        route = state.get('route')
        if route == 'rag' or (hasattr(route, 'router') and route.router == 'rag'):
            return {"trace": state.get("trace", [])}
        print(f"Executing SQL: {state.get('sql', '')}")
        executor = SQLiteTool("data/northwind.sqlite")
        result = executor.execute_query(state.get("sql", ""))
        # state["sql_result"] = result
        state.setdefault("trace", []).append(f"Executor → success={result.get('success')}")
        return {"sql_result": result, "trace": state["trace"]}


    def _node_synthesizer(self,state: AgentState):
        # normalize doc source: prefer 'docs' then 'retrieved_docs'
        docs = state.get("docs", state.get("retrieved_docs", []))
        # forward requested format hint if present so synthesizer can enforce exact types
        format_hint = state.get("format_hint", "")
        
        # For RAG route, we don't need to generate SQL
        route = state.get('route')
        sql_result = state.get("sql_result", {})
        
        # Generate final answer using the SynthesizerGenerator
        final_answer, citations, llmconfidence = self.synthesizer(
            question=state["question"],
            sql_result=sql_result,
            docs=docs,
            format_hint=format_hint
        )

        # Calculate confidence using custom method
        confidence = self._calculate_confidence(state)
        # state["final_answer"] = final_answer
        # state["citations"] = citations
        # state["confidence"] = confidence
        state.setdefault("trace", []).append(f"Synthesizer → answer generated")
        print(f"Final Answer: {final_answer}, Citations: {citations}, Confidence: {confidence}")
        
        # Only include sql in the return value if it's already in the state
        # This prevents the synthesizer from generating SQL for RAG routes
        result = {
            "final_answer": final_answer,
            "citations": citations,
            "confidence": confidence,
            "trace": state["trace"],
        }
        
        return result


    def _node_repair(self,state: AgentState):
        """Retry failed SQL up to 2 times."""
        # If route is 'rag', no need to repair SQL
        route = state.get('route')
        if route == 'rag' or (hasattr(route, 'router') and route.router == 'rag'):
            return {"trace": state.get("trace", [])}
            
        if state.get("sql_result", {}).get("success"):
            # nothing to do
            return {}
        attempts = state.get("repair_attempts", 0)
        print(f"New blah failed, attempt {attempts} to repair...")
        if attempts < 2:
            state["repair_attempts"] = attempts + 1
            g = state.get("repair_attempts", 0)
            print(f"Repairing SQL, attempt G {g}...")
            state.setdefault("trace", []).append("Repairing SQL query...")
            # regenerate SQL using NL2SQL node logic (it returns only changed keys)
            return self._node_nl2sql(state)
        else:
            state["repair_attempts"] = attempts
            state.setdefault("trace", []).append("Repair failed after 2 attempts")
            return {"trace": state["trace"]}


    def _checkpointer_node(self,state: AgentState):
        # Log current state snapshot
        print("TRACE:", dict(state))
        # Propagate only a small set of keys to avoid concurrent full-state writes,
        # but ensure important outputs are returned from the graph.
        keys_to_propagate = ["final_answer", "citations", "confidence", "trace", "sql", "sql_result"]
        out = {k: state[k] for k in keys_to_propagate if k in state}
        return out

    def _calculate_confidence(self, state: AgentState) -> float:
        """
        Calculate confidence score based on multiple factors:
        - Retrieval score coverage
        - SQL success
        - Non-empty rows
        - Repair penalty
        
        Returns a confidence score between 0 and 1.
        """
        base_confidence = 0.5  # Start with a neutral confidence
        factors = []
        
        # 1. Retrieval score coverage factor
        docs = state.get("docs", state.get("retrieved_docs", []))
        if docs:
            # Calculate average relevance score of top documents
            scores = [d.get("score", 0) for d in docs if isinstance(d, dict) and "score" in d]
            if scores:
                # Normalize scores to 0-1 range (assuming scores are typically 0-10)
                avg_score = sum(scores) / len(scores)
                norm_score = min(avg_score / 10.0, 1.0)  # Normalize to 0-1
                factors.append(("retrieval_coverage", norm_score, 0.3))  # 30% weight
        
        # 2. SQL success factor (for SQL routes only)
        route = state.get('route')
        if route == 'sql' or (hasattr(route, 'router') and route.router == 'sql'):
            sql_result = state.get("sql_result", {})
            sql_success = 1.0 if sql_result.get("success", False) else 0.5
            factors.append(("sql_success", sql_success, 0.3))  # 30% weight
            
            # 3. Non-empty rows factor (for SQL routes only)
            rows = sql_result.get("rows", [])
            has_rows = 1.0 if rows and len(rows) > 0 else 0.7
            factors.append(("non_empty_rows", has_rows, 0.2))  # 20% weight
        else:
            # For RAG routes, we give full weight to retrieval quality
            factors.append(("rag_route", 1.0, 0.5))  # 50% weight
        
        # 4. Repair penalty factor
        repair_attempts = state.get("repair_attempts", 0)
        if repair_attempts > 0:
            # Apply penalty for each repair attempt (0.8^attempts)
            repair_factor = 0.8 ** repair_attempts
            factors.append(("repair_penalty", repair_factor, 0.2))  # 20% weight
        else:
            factors.append(("no_repairs", 1.0, 0.2))  # 20% weight
        
        # Calculate weighted average of all factors
        if factors:
            weighted_sum = sum(score * weight for _, score, weight in factors)
            total_weight = sum(weight for _, _, weight in factors)
            confidence = weighted_sum / total_weight if total_weight > 0 else base_confidence
        else:
            confidence = base_confidence
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        # Log the factors for debugging
        factor_str = ", ".join([f"{name}={score:.2f}*{weight:.2f}" for name, score, weight in factors])
        print(f"Confidence calculation: {factor_str} = {confidence:.2f}")
        
        return confidence

    ### CONDITIONAL EDGES ###

    def decide_after_retriever(self, state: AgentState) -> str:
        route = state.get('route')
        # Handle both string and DSPy Prediction object
        if route == 'rag' or (hasattr(route, 'router') and route.router == 'rag'):
            return 'synthesizer'
        return 'planner'

    def decide_sql_repair(self, state: AgentState) -> str:
        # If route is 'rag', no need to repair SQL
        route = state.get('route')
        if route == 'rag' or (hasattr(route, 'router') and route.router == 'rag'):
            return "END"
            
        # If SQL failed and we have repair attempts left, go to repair, else end
        success = state.get("sql_result", {}).get("success", True)
        attempts = state.get("repair_attempts", 0)
        print(f"SQL success: {success}, Repair attempts: {attempts}")
        if not success and attempts < 2:
            return "repair"
        return "END"

    # ------------------------------
    # Build Graph
    # ------------------------------
    def _build_graph(self):
        sg = StateGraph(self.AgentState)

        sg.add_node("router", self._node_router)
        sg.add_node("retriever", self._node_retriever)
        sg.add_node("planner", self._node_planner)
        sg.add_node("nl2sql", self._node_nl2sql)
        sg.add_node("executor", self._node_executor)
        sg.add_node("synthesizer", self._node_synthesizer)
        sg.add_node("repair", self._node_repair)
        sg.add_node("checkpointer", self._checkpointer_node)

        sg.set_entry_point("router")


        sg.add_edge("router", "retriever") # Router → Retriever ( - sql | hybrid | rag)

        # Retriever → Synthesizer (if rag), else → Planner
        sg.add_conditional_edges(
            "retriever",
            self.decide_after_retriever,
            {"synthesizer": "synthesizer", "planner": "planner"}
        )

        # Planner → NL2SQL
        sg.add_edge("planner", "nl2sql")
        # NL2SQL → Executor
        sg.add_edge("nl2sql", "executor")
        # Executor → Synthesizer
        sg.add_edge("executor", "synthesizer")

        # Synthesizer → Repair (if invalid), else END
        sg.add_conditional_edges(
            "synthesizer",
            self.decide_sql_repair,
            {"repair": "repair", "END": END}
        )

        # Repair → NL2SQL (if SQL failed) OR Synthesizer (if format failed)
        sg.add_conditional_edges(
            "repair",
            lambda state: "nl2sql" if state.get("sql_error") else "synthesizer",
            {"nl2sql": "nl2sql", "synthesizer": "synthesizer"}
        )

        # Checkpointer after each node (checkpointer returns empty -> no concurrent writes)
        # Avoid attaching checkpointer as a terminating branch from every node.
        # Doing so caused the graph to take the checkpointer->END path early (e.g., after router)
        # and prevented downstream nodes from running. Instead attach the checkpointer
        # at the end of the main flow so it can capture the final snapshot without
        # interrupting execution.
        sg.add_edge("synthesizer", "checkpointer")
        sg.add_edge("repair", "checkpointer")
        sg.add_edge("checkpointer", END)

        app = sg.compile()
        image_bytes = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
        # To save it to a file:
        with open("graph_structure.png", "wb") as f:
            f.write(image_bytes)

        return sg.compile()


    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Simple heuristic to collect DB table names referenced in SQL."""
        if not sql:
            return []
        sql_low = sql.lower()
        tables = []
        for name in ["orders", '"order details"', "order details", "products", "customers", "categories", "shippers", "suppliers"]:
            if name in sql_low:
                # normalize to canonical names used in contract (cap-sensitive where applicable)
                if "order details" in name:
                    tables.append("Order Details")
                else:
                    tables.append(name.capitalize() if name.islower() else name)
        # dedupe preserving order
        seen = set()
        out = []
        for t in tables:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _coerce_by_format(self, value, format_hint: str, sql_result: Dict[str, Any]=None, docs: List[Dict[str, Any]]=None, question: str = None):
        """Coerce final_answer to the requested format_hint when possible.
        Prefer sql_result rows, then provided value, then docs as fallback.
        Uses the question to prefer more relevant doc chunks when extracting numbers.
        """
        import re
        def extract_number_from_text(text):
            m = re.search(r"[-+]?\d*\.?\d+", text)
            return float(m.group(0)) if m else None

        if format_hint is None:
            return value

        fmt = format_hint.strip()

        # Helper to coerce single primitive
        def to_int(v):
            try:
                return int(float(v))
            except Exception:
                return 0
        def to_float(v):
            try:
                return round(float(v), 2)
            except Exception:
                return 0.0

        # Use SQL first if present
        rows = (sql_result or {}).get("rows") if sql_result else None
        cols = (sql_result or {}).get("columns") if sql_result else None

        try:
            # INT
            if fmt == "int":
                # try sql first
                if rows:
                    r = rows[0][0] if len(rows[0])>0 else None
                    return to_int(r)
                # try provided value
                if value is not None:
                    # Check if the value is a string that contains a number
                    if isinstance(value, str) and ":" in value:
                        # Extract the number after the colon
                        parts = value.split(":")
                        if len(parts) > 1 and parts[1].strip().isdigit():
                            return to_int(parts[1].strip())
                    
                    if isinstance(value, (list,tuple)):
                        candidate = value[0][0] if value and isinstance(value[0], (list,tuple)) else value[0]
                    else:
                        candidate = value
                    if isinstance(candidate, (int,float,str)):
                        return to_int(candidate)
                # try docs (prefer most relevant doc by question keywords)
                if docs:
                    # compute question tokens
                    q = (question or "").lower()
                    q_tokens = set([t for t in re.findall(r'\w+', q) if len(t) > 3])
                    best_doc = None
                    best_score = -1
                    for d in docs:
                        txt = d.get("content","") if isinstance(d, dict) else str(d)
                        txt_low = (txt or "").lower()
                        # simple relevance: count occurrences of question tokens
                        score = sum(txt_low.count(tok) for tok in q_tokens) if q_tokens else 0
                        # also boost if doc explicitly contains terms like 'beverage'/'beverages' and question mentions 'beverage'
                        # if "beverage" in txt_low and "beverage" in q:
                        #     score += 5
                        if score > best_score:
                            best_score = score
                            best_doc = txt
                    # try extracting from best_doc first
                    if best_doc:
                        # Look specifically for "Beverages unopened: 14 days" pattern if question is about beverages
                        # if "beverage" in q.lower() and "unopened" in q.lower():
                        #     beverage_match = re.search(r"Beverages\s+unopened:\s+(\d+)\s+days", best_doc)
                        #     if beverage_match:
                        #         return to_int(beverage_match.group(1))
                        
                        num = extract_number_from_text(best_doc or "")
                        if num is not None:
                            return to_int(num)
                    # fallback to scanning docs in order (legacy behavior)
                    for d in docs:
                        txt = d.get("content") if isinstance(d, dict) else str(d)
                        # Look specifically for "Beverages unopened: 14 days" pattern if question is about beverages
                        # if "beverage" in q.lower() and "unopened" in q.lower():
                        #     beverage_match = re.search(r"Beverages\s+unopened:\s+(\d+)\s+days", txt)
                        #     if beverage_match:
                        #         return to_int(beverage_match.group(1))
                                
                        num = extract_number_from_text(txt or "")
                        if num is not None:
                            return to_int(num)
                return 0

            # FLOAT
            if fmt == "float":
                if rows:
                    r = rows[0][0] if len(rows[0])>0 else None
                    return to_float(r)
                if value is not None:
                    if isinstance(value, (list,tuple)):
                        candidate = value[0][0] if value and isinstance(value[0], (list,tuple)) else value[0]
                    else:
                        candidate = value
                    if isinstance(candidate, (int,float,str)):
                        return to_float(candidate)
                # prefer best doc by question when extracting floats too
                if docs:
                    q = (question or "").lower()
                    q_tokens = set([t for t in re.findall(r'\w+', q) if len(t) > 3])
                    best_doc = None
                    best_score = -1
                    for d in docs:
                        txt = d.get("content","") if isinstance(d, dict) else str(d)
                        txt_low = (txt or "").lower()
                        score = sum(txt_low.count(tok) for tok in q_tokens) if q_tokens else 0
                        # if "beverage" in txt_low and "beverage" in q:
                        #     score += 5
                        if score > best_score:
                            best_score = score
                            best_doc = txt
                    if best_doc:
                        num = extract_number_from_text(best_doc or "")
                        if num is not None:
                            return to_float(num)
                    for d in docs:
                        txt = d.get("content") if isinstance(d, dict) else str(d)
                        num = extract_number_from_text(txt or "")
                        if num is not None:
                            return to_float(num)
                return 0.0

            # OBJECT like "{category:str, quantity:int}"
            if fmt.startswith("{") and fmt.endswith("}"):
                # parse keys and types
                body = fmt[1:-1]
                parts = [p.strip() for p in body.split(",")]
                spec = []
                for p in parts:
                    if ":" in p:
                        k,t = [x.strip() for x in p.split(":",1)]
                        spec.append((k,t))
                obj = {}
                # Use SQL mapping if available
                if rows and cols:
                    first = rows[0]
                    # try to map by index -> spec order; if column name matches key use that
                    col_map = {c.lower(): i for i,c in enumerate(cols)}
                    for i,(k,t) in enumerate(spec):
                        # prefer column with same name
                        if k.lower() in col_map:
                            v = first[col_map[k.lower()]]
                        elif i < len(first):
                            v = first[i]
                        else:
                            v = None
                        if t.startswith("int"):
                            obj[k] = to_int(v)
                        elif t.startswith("float"):
                            obj[k] = to_float(v)
                        else:
                            obj[k] = str(v) if v is not None else ""
                    return obj
                # otherwise attempt from value (if dict)
                if isinstance(value, dict):
                    for k,t in spec:
                        v = value.get(k) or value.get(k.lower())
                        if t.startswith("int"):
                            obj[k] = to_int(v)
                        elif t.startswith("float"):
                            obj[k] = to_float(v)
                        else:
                            obj[k] = str(v) if v is not None else ""
                    return obj
                # lastly heuristics from docs
                text = ""
                if docs:
                    text = " ".join([d.get("content","") if isinstance(d, dict) else str(d) for d in docs])
                for k,t in spec:
                    if t.startswith("int"):
                        num = extract_number_from_text(text or "")
                        obj[k] = to_int(num) if num is not None else 0
                    elif t.startswith("float"):
                        num = extract_number_from_text(text or "")
                        obj[k] = to_float(num) if num is not None else 0.0
                    else:
                        # take first capitalized token as category candidate
                        m = re.search(r"\b([A-Z][a-zA-Z0-9&\-\s]{0,50})\b", text or "")
                        obj[k] = m.group(0).strip() if m else ""
                return obj

            # LIST hints like "list[{product:str, revenue:float}]"
            if fmt.startswith("list"):
                # if sql rows present, map rows -> list of dicts using columns
                if rows and cols:
                    out_list = []
                    # if inner is object spec, try to build objects using columns
                    inner = None
                    if "{" in fmt and "}" in fmt:
                        inner = fmt[fmt.index("{"):fmt.index("}")+1]
                        # parse inner spec
                        inner_body = inner[1:-1]
                        parts = [p.strip() for p in inner_body.split(",")]
                        spec = []
                        for p in parts:
                            if ":" in p:
                                k,t = [x.strip() for x in p.split(":",1)]
                                spec.append((k,t))
                        for row in rows:
                            row_obj = {}
                            for i,(k,t) in enumerate(spec):
                                # prefer column name match
                                v = None
                                for ci,col in enumerate(cols):
                                    if col.lower() == k.lower():
                                        v = row[ci]
                                        break
                                if v is None and i < len(row):
                                    v = row[i]
                                if t.startswith("int"):
                                    row_obj[k] = to_int(v)
                                elif t.startswith("float"):
                                    row_obj[k] = to_float(v)
                                else:
                                    row_obj[k] = str(v) if v is not None else ""
                            out_list.append(row_obj)
                        return out_list
                    # otherwise just return rows as list
                    return rows
                # fallback: no SQL -> return empty list
                return []

            # default: return value as-is
            return value
        except Exception:
            return value

    def _build_output_contract(self, q_input, state: AgentState):
        """Construct the required Output Contract dict."""
        # q_input is a dict with id/question/format_hint
        qid = q_input.get("id", "")
        qtext = q_input.get("question", q_input.get("text", ""))
        fmt = q_input.get("format_hint")

        final_answer = state.get("final_answer")
        print("Final answer before coercion:", final_answer, "with format hint:", fmt)
        sql = state.get("sql", "") or ""
        confidence = float(state.get("confidence", 0.0))
        sql_res = state.get("sql_result", {}) or {}
        docs = state.get("docs", []) or []

        # If synthesizer returned a doc string but requested format is not a string,
        # prefer to coerce from SQL rows or docs rather than pass raw doc text.
        coerced = self._coerce_by_format(final_answer, fmt, sql_result=sql_res, docs=docs, question=qtext)

        # If coercion produced None or an incompatible type, fallback to stricter attempts:
        if coerced is None:
            # try again using SQL directly
            coerced = self._coerce_by_format(sql_res.get("rows"), fmt, sql_result=sql_res, docs=docs, question=qtext)
        # Final safety: ensure type matches fmt
        # (for numeric/object/list we already produce correct types in _coerce_by_format)
        # explanation: short two-sentence style explanation
        explanation_parts = []
        if sql:
            explanation_parts.append(f"Generated SQL used {', '.join(self._extract_tables_from_sql(sql)) or 'no DB tables detected'};")
        if coerced in (None, "No answer found."):
            explanation_parts.append("No structured answer could be produced; returned default.")
        else:
            explanation_parts.append("Answer derived and coerced to requested format.")
        explanation = " ".join(explanation_parts)[:500]  # keep short

        # citations: DB tables + doc chunk ids
        citations = []
        # add tables from SQL
        citations.extend(self._extract_tables_from_sql(sql))
        # add doc chunk ids if available
        for d in docs:
            if isinstance(d, dict):
                if "chunk_id" in d:
                    citations.append(f"{d.get('source','docs')}::"+str(d["chunk_id"]))
                elif "id" in d:
                    citations.append(f"{d.get('source','docs')}::"+str(d["id"]))
        # dedupe
        seen = set()
        citations = [c for c in citations if not (c in seen or seen.add(c))]

        return {
            "id": qid,
            "final_answer": final_answer,
            "sql": sql,
            "confidence": confidence,
            "explanation": explanation,
            "citations": citations
        }

    def run(self, question):
        # accept either:
        #  - dict with id/question/format_hint (existing behavior), or
        #  - raw question string (backwards-compatible callers)
        if isinstance(question, dict):
            q_input = question
            q_text = q_input.get("question", q_input.get("text", ""))
        else:
            # synthesize a minimal input contract for string callers
            q_input = {"id": "", "question": str(question), "format_hint": None}
            q_text = str(question)

        initial_state = self.AgentState(question=q_text)
        # carry forward format_hint into state for synthesizer to use (if provided)
        if isinstance(q_input, dict) and q_input.get("format_hint") is not None:
            initial_state["format_hint"] = q_input.get("format_hint")
        initial_state["trace"] = []
        final_state = self.graph.invoke(initial_state)
        print("Final State:", dict(final_state))
        # Build and return the Output Contract (never raise)
        out = self._build_output_contract(q_input, final_state)
        return out

    def run_router_demo(self, seed: int = 0):
        """
        Quick helper to run the small DSPy-style Router demo (train_demo).
        Call this to see baseline vs trained accuracy on a tiny handcrafted split.
        """
        if hasattr(self.router, "train_demo"):
            return self.router.train_demo(seed=seed)
        return {"error": "router has no train_demo"}

# if __name__ == "__main__":
#     graph = GraphHybrid()
#     # demonstrate both usages: dict with format_hint to ensure coercion
#     test_input = {
#         "id": "rag_policy_beverages_return_days",
#         "question": "According to the product policy, what is the return window (days) for unopened Beverages? Return an integer.",
#         "format_hint": "int"
#     }
#     result = graph.run(test_input)
#     print("\nFinal Output Contract:\n", result)
