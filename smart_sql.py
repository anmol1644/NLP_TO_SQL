import os
import time
import hashlib
import json
import re
from typing import Dict, List, Optional, Tuple, Any, Union

import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema.memory import BaseMemory

from db_analyzer import DatabaseAnalyzer


class SmartSQLGenerator:
    """
    AI-powered SQL query generator that works with any PostgreSQL database
    without relying on predefined templates
    """
    
    def __init__(
        self,
        db_analyzer: DatabaseAnalyzer,
        model_name: str = "gemini-2.0-flash",
        use_cache: bool = True,
        cache_file: str = "query_cache.json",
        use_memory: bool = True,
        memory_persist_dir: str = "./memory_store"
    ):
        """
        Initialize the SQL generator
        
        Args:
            db_analyzer: Database analyzer instance
            model_name: Generative AI model to use
            use_cache: Whether to cache query results
            cache_file: Path to the query cache file
            use_memory: Whether to use conversation memory
            memory_persist_dir: Directory to persist memory embeddings
        """
        load_dotenv()
        self.db_analyzer = db_analyzer
        self.model_name = model_name
        self.use_cache = use_cache
        self.cache_file = cache_file
        self.cache = self._load_cache() if use_cache else {}
        self.use_memory = use_memory
        
        # Session-specific memory for tracking conversation context
        self.session_context = {
            "user_info": {},
            "query_sequence": [],
            "important_values": {},
            "last_query_result": None,
            "entity_mentions": {}
        }
        
        # Initialize the AI model
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Configure Google Gemini
        genai.configure(api_key=api_key)
        
        # Initialize LangChain with Gemini
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0, 
            google_api_key=api_key
        )
        
        # Initialize memory system if enabled
        self.memory = None
        if use_memory:
            self.memory = self._initialize_memory(memory_persist_dir)
            
        # Define SQL generation prompt with memory context
        memory_var = "{memory}\n\n" if use_memory else ""
        self.sql_prompt = PromptTemplate(
            input_variables=["schema", "question", "examples"] + (["memory"] if use_memory else []),
            template=f"""You are an expert SQL developer specializing in PostgreSQL databases. Your job is to translate natural language questions into precise and efficient SQL queries.

{memory_var}### DATABASE SCHEMA:
{{schema}}

### EXAMPLES OF GOOD SQL PATTERNS:
{{examples}}

### TASK:
Convert the following question into a single PostgreSQL SQL query:
"{{question}}"

### GUIDELINES:
1. Create only PostgreSQL-compatible SQL
2. Focus on writing efficient queries
3. Use proper table aliases for clarity
4. Include appropriate JOINs based on database relationships
5. Include comments explaining complex parts of your query
6. Always use proper column quoting with double quotes for columns with spaces or special characters
7. NEVER use any placeholder values in your final query
8. Use any available user information (name, role, IDs) from memory to personalize the query if applicable
9. Use specific values from previous query results when referenced (e.g., "this product", "these customers", "that date")
10. For follow-up questions or refinements, maintain the filters and conditions from the previous query
11. If the follow-up question is only changing which columns to display, KEEP ALL WHERE CONDITIONS from the previous query
12. When user asks for "this" or refers to previous results implicitly, use the context from the previous query
13. When user refers to "those" or "these" results with terms like "highest" or "lowest", ONLY consider the exact rows from the previous result set, NOT the entire table
14. If IDs from previous results are provided in the memory context, use them in a WHERE clause to limit exactly to those rows
15. Only those tables must be joined that have a foreign key relationship with the table being queried

### OUTPUT FORMAT:
Provide ONLY the SQL query with no additional text, explanation, or markdown formatting.
"""
        )
        
        # Create the SQL generation chain
        self.sql_chain = LLMChain(
            llm=self.llm,
            prompt=self.sql_prompt
        )
        
        # Define validation prompt
        self.validation_prompt = PromptTemplate(
            input_variables=["schema", "sql", "error"] + (["memory"] if use_memory else []),
            template=f"""You are an expert SQL developer specializing in PostgreSQL databases. Your job is to fix SQL query errors.

{memory_var}### DATABASE SCHEMA:
{{schema}}

### QUERY WITH ERROR:
```sql
{{sql}}
```

### ERROR MESSAGE:
{{error}}

### TASK:
Fix the SQL query to resolve the error.

### GUIDELINES:
1. Create only PostgreSQL-compatible SQL
2. Maintain the original query intent
3. Fix any syntax errors, typos, or invalid column references
4. NEVER use any placeholder values in your final query
5. Use any available user information (name, role, IDs) from memory to personalize the query if applicable

### OUTPUT FORMAT:
Provide ONLY the corrected SQL query with no additional text, explanation, or markdown formatting.
"""
        )
        
        # Create the validation chain
        self.validation_chain = LLMChain(
            llm=self.llm,
            prompt=self.validation_prompt
        )
        
        # Prepare schema context
        self.schema_context = None
        self.example_patterns = None
    
    def _initialize_memory(self, persist_dir: str) -> Optional[BaseMemory]:
        """Initialize vector store memory with Gemini embeddings"""
        try:
            # Ensure the directory exists
            os.makedirs(persist_dir, exist_ok=True)
            
            # Initialize embeddings with the specified model
            embedding_function = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-exp-03-07",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            
            # Create or load the vector store
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding_function,
                collection_name="sql_conversation_memory"
            )
            
            # Create the retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Create memory
            memory = VectorStoreRetrieverMemory(
                retriever=retriever,
                memory_key="memory",
                return_docs=True,
                input_key="question",
                # Memory types for retriever
                input_prefix="RETRIEVAL_QUERY: ",
                document_prefix="RETRIEVAL_DOCUMENT: "
            )
            
            return memory
        except Exception as e:
            print(f"Error initializing memory: {e}")
            return None
    
    def _store_in_memory(self, question: str, sql: str, results: Any = None) -> None:
        """Store the question, generated SQL and results in memory"""
        if not self.memory or not self.use_memory:
            return
            
        try:
            # Create document with question and SQL
            content = f"Question: {question}\nSQL: {sql}"
            
            # Extract and store personal information
            personal_info = self._extract_personal_info(question, results)
            if personal_info:
                content = f"{personal_info}\n\n{content}"
            
            # Add result summary if available
            if results:
                try:
                    # Count rows or summarize results
                    if isinstance(results, list) and results:
                        num_rows = len(results)
                        sample = results[0] if results else {}
                        columns = list(sample.keys()) if isinstance(sample, dict) else []
                        result_summary = f"\nReturned {num_rows} rows with columns: {', '.join(columns)}"
                        
                        # Include sample results (first 3 rows at most)
                        if num_rows > 0:
                            result_summary += "\nSample results:"
                            for i, row in enumerate(results[:3]):
                                result_summary += f"\nRow {i+1}: {str(row)}"
                        
                        content += result_summary
                except Exception as e:
                    print(f"Error summarizing results: {e}")
                
            # Store in memory
            self.memory.save_context(
                {"question": question}, 
                {"memory": content}
            )
        except Exception as e:
            print(f"Error storing in memory: {e}")

    def _extract_personal_info(self, question: str, results: Any = None) -> str:
        """Extract personal information from user queries or results"""
        personal_info = []
        
        # Check for name information
        name_patterns = [
            r"my name is (?P<name>[\w\s]+)",
            r"I am (?P<name>[\w\s]+)",
            r"I'm (?P<name>[\w\s]+)",
            r"call me (?P<name>[\w\s]+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                name = match.group("name").strip()
                personal_info.append(f"User name: {name}")
                break
        
        # Check for other personal identifiers in the question
        if "my" in question.lower():
            id_patterns = [
                r"my (?P<id_type>user|customer|employee|sales|account|order|client|supplier|vendor) (?P<id_value>\w+)",
                r"my (?P<id_type>user|customer|employee|sales|account|order|client|supplier|vendor) id is (?P<id_value>\w+)",
                r"my (?P<id_type>user|customer|employee|sales|account|order|client|supplier|vendor) number is (?P<id_value>\w+)",
            ]
            
            for pattern in id_patterns:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    id_type = match.group("id_type").strip()
                    id_value = match.group("id_value").strip()
                    personal_info.append(f"User {id_type} ID: {id_value}")
                    break
        
        # Check for personal context "I am a X"
        role_patterns = [
            r"I am a (?P<role>[\w\s]+)",
            r"I'm a (?P<role>[\w\s]+)",
            r"I work as a (?P<role>[\w\s]+)",
            r"my role is (?P<role>[\w\s]+)"
        ]
        
        for pattern in role_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                role = match.group("role").strip()
                personal_info.append(f"User role: {role}")
                break
        
        if personal_info:
            return "\n".join(personal_info)
        return ""

    def _get_memory_context(self, question: str) -> str:
        """Retrieve relevant context from memory for a question"""
        if not self.memory or not self.use_memory:
            return ""
            
        try:
            # Get relevant memories
            memory_context = self.memory.load_memory_variables({"question": question})
            return memory_context.get("memory", "")
        except Exception as e:
            print(f"Error retrieving from memory: {e}")
            return ""
    
    def _load_cache(self) -> Dict:
        """Load query cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_cache(self) -> None:
        """Save query cache to disk"""
        if self.use_cache:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
    
    def _get_question_hash(self, question: str) -> str:
        """Generate a hash for a question to use as cache key"""
        # Normalize the question: lowercase, remove punctuation, extra spaces
        normalized = re.sub(r'[^\w\s]', '', question.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _prepare_schema_context(self) -> None:
        """Prepare the database schema context for the AI model"""
        if not self.schema_context:
            # Get rich schema information for context
            self.schema_context = self.db_analyzer.get_rich_schema_context()
            
            # Generate example query patterns based on the schema
            self.example_patterns = self._generate_example_patterns()
    
    def _generate_example_patterns(self) -> str:
        """
        Generate example SQL patterns based on the analyzed schema
        
        Returns:
            String with example SQL patterns
        """
        if not self.db_analyzer.schema_info:
            self.db_analyzer.analyze_schema()
        
        schema = self.db_analyzer.schema_info
        examples = []
        
        # Get table names for examples
        table_names = list(schema["tables"].keys())
        if not table_names:
            return "No tables found in the database."
        
        # Find tables with time-related columns for time series examples
        time_tables = []
        for table_name, table_info in schema["tables"].items():
            for col in table_info["columns"]:
                if "date" in col["type"].lower() or "time" in col["type"].lower():
                    time_tables.append((table_name, col["name"]))
                    break
        
        # Example 1: Basic query with filtering
        if table_names:
            table = schema["tables"][table_names[0]]
            if table["columns"]:
                col1 = table["columns"][0]["name"]
                filter_col = None
                
                # Find a good column to filter on
                for col in table["columns"]:
                    if any(t in col["type"].lower() for t in ["varchar", "text", "char"]):
                        filter_col = col["name"]
                        break
                
                if not filter_col and len(table["columns"]) > 1:
                    filter_col = table["columns"][1]["name"]
                
                if filter_col:
                    examples.append(f"""Example 1: Simple filtering
Query: "Show me all {table_names[0]} where {filter_col} contains 'example'"
SQL:
```sql
SELECT * 
FROM "{table_names[0]}"
WHERE "{filter_col}" LIKE '%example%'
LIMIT 10;
```
""")
        
        # Example 2: Aggregation
        if table_names:
            table = schema["tables"][table_names[0]]
            numeric_col = None
            group_col = None
            
            # Find numeric and categorical columns
            for col in table["columns"]:
                if any(t in col["type"].lower() for t in ["int", "float", "numeric", "decimal"]):
                    numeric_col = col["name"]
                elif any(t in col["type"].lower() for t in ["varchar", "text", "char"]) and not group_col:
                    group_col = col["name"]
            
            if numeric_col and group_col:
                examples.append(f"""Example 2: Aggregation with grouping
Query: "Calculate the total {numeric_col} grouped by {group_col}"
SQL:
```sql
SELECT "{group_col}", SUM("{numeric_col}") as total_{numeric_col}
FROM "{table_names[0]}"
GROUP BY "{group_col}"
ORDER BY total_{numeric_col} DESC
LIMIT 10;
```
""")
        
        # Example 3: Joins between related tables
        if len(schema["relationships"]) > 0:
            rel = schema["relationships"][0]
            src_table = rel["source_table"]
            tgt_table = rel["target_table"]
            src_col = rel["source_columns"][0]
            tgt_col = rel["target_columns"][0]
            
            # Find a column from each table to display
            src_display_col = next((col["name"] for col in schema["tables"][src_table]["columns"] 
                                  if col["name"] != src_col and not col["name"].endswith("_id")), src_col)
            tgt_display_col = next((col["name"] for col in schema["tables"][tgt_table]["columns"] 
                                  if col["name"] != tgt_col and not col["name"].endswith("_id")), tgt_col)
            
            examples.append(f"""Example 3: Joining related tables
Query: "Show {src_table} with their related {tgt_table}"
SQL:
```sql
SELECT s."{src_display_col}", t."{tgt_display_col}"
FROM "{src_table}" s
JOIN "{tgt_table}" t ON s."{src_col}" = t."{tgt_col}"
LIMIT 10;
```
""")
        
        # Example 4: Time series analysis
        if time_tables:
            table_name, date_col = time_tables[0]
            
            # Find a numeric column to aggregate
            numeric_col = None
            for col in schema["tables"][table_name]["columns"]:
                if any(t in col["type"].lower() for t in ["int", "float", "numeric", "decimal"]):
                    numeric_col = col["name"]
                    break
            
            if numeric_col:
                examples.append(f"""Example 4: Time series analysis
Query: "Show monthly totals of {numeric_col} in {table_name}"
SQL:
```sql
SELECT 
    DATE_TRUNC('month', "{date_col}") AS month,
    SUM("{numeric_col}") AS total_{numeric_col}
FROM "{table_name}"
GROUP BY DATE_TRUNC('month', "{date_col}")
ORDER BY month DESC;
```
""")
        
        # Example 5: Subqueries and complex filtering
        if table_names and len(table_names) > 1:
            table1 = table_names[0]
            table2 = table_names[1]
            
            # Find primary keys
            pk1 = next((col["name"] for col in schema["tables"][table1]["columns"] if col.get("primary_key")), None)
            
            if pk1:
                examples.append(f"""Example 5: Subqueries
Query: "Find {table1} that have more than 5 associated {table2}"
SQL:
```sql
SELECT t1.*
FROM "{table1}" t1
WHERE (
    SELECT COUNT(*)
    FROM "{table2}" t2
    WHERE t2."{table1}_{pk1}" = t1."{pk1}"
) > 5;
```
""")
        
        return "\n".join(examples)
    
    def _validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate the SQL query for basic safety and syntax
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for dangerous operations
        dangerous_patterns = [
            r'\bDROP\b\s+(?:\bDATABASE\b|\bSCHEMA\b)',
            r'\bTRUNCATE\b',
            r'\bDELETE\b\s+FROM\b\s+\w+\s*(?!\bWHERE\b)',  # DELETE without WHERE
            r'\bUPDATE\b\s+\w+\s+SET\b\s+(?:\w+\s*=\s*\w+)(?:\s*,\s*\w+\s*=\s*\w+)*\s*(?!\bWHERE\b)',  # UPDATE without WHERE
            r'\bCREATE\b\s+(?:\bDATABASE\b|\bSCHEMA\b)',
            r'\bDROP\b\s+\bTABLE\b',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return False, f"Query contains potentially dangerous operation: {pattern}"
        
        # Basic syntax check
        if not re.search(r'\bSELECT\b', sql, re.IGNORECASE):
            return False, "Only SELECT queries are allowed"
        
        # Check for unresolved placeholders
        if re.search(r'\{.*?\}', sql):
            return False, "Query contains unresolved placeholders"
        
        return True, None
    
    def _analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze the question to extract key information
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary with analysis results
        """
        # TODO: Implement more sophisticated question analysis
        # For now, just return basic info
        analysis = {
            "complexity": "simple" if len(question.split()) < 10 else "complex",
            "keywords": [],
        }
        
        # Extract key business intelligence terms
        bi_terms = [
            "total", "average", "count", "sum", "group by", "trend",
            "compare", "ratio", "percentage", "growth", "rank", 
            "top", "bottom", "highest", "lowest"
        ]
        
        for term in bi_terms:
            if term in question.lower():
                analysis["keywords"].append(term)
        
        return analysis
    
    def generate_sql(self, question: str) -> Dict[str, Any]:
        """
        Generate a SQL query from a natural language question
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary with query generation details
        """
        start_time = time.time()
        
        # Check cache first if enabled
        if self.use_cache:
            question_hash = self._get_question_hash(question)
            if question_hash in self.cache:
                cached_result = self.cache[question_hash].copy()
                cached_result["source"] = f"{cached_result['source']} (cached)"
                cached_result["execution_time"] = 0
                return cached_result
        
        # Prepare schema context if not already prepared
        self._prepare_schema_context()
        
        # Analyze the question
        question_analysis = self._analyze_question(question)
        
        try:
            # Prepare params for SQL generation
            params = {
                "schema": self.schema_context,
                "question": question,
                "examples": self.example_patterns
            }
            
            # Add memory if enabled
            if self.use_memory:
                memory_context = self._prepare_memory_for_query(question) or ""
                params["memory"] = memory_context
                
            # Generate SQL with LangChain + Gemini
            response = self.sql_chain.invoke(params)
            
            if isinstance(response, dict) and "text" in response:
                sql_response = response["text"]
            else:
                return {
                    "success": False,
                    "sql": None,
                    "source": "ai",
                    "confidence": 0,
                    "error": "AI returned an unexpected response format",
                    "execution_time": time.time() - start_time
                }
            
            # Clean up response to extract just the SQL
            sql = sql_response.strip()
            if sql.startswith('```') and sql.endswith('```'):
                sql = sql[3:-3].strip()
            if sql.startswith('sql') or sql.startswith('SQL'):
                sql = sql[3:].strip()
            
            # Validate SQL
            is_valid, error = self._validate_sql(sql)
            if not is_valid:
                return {
                    "success": False,
                    "sql": sql,
                    "source": "ai",
                    "confidence": 50,
                    "error": error,
                    "execution_time": time.time() - start_time
                }
            
            result = {
                "success": True,
                "sql": sql,
                "source": "ai",
                "confidence": 90,
                "execution_time": time.time() - start_time
            }
            
            # Cache the result if caching is enabled
            if self.use_cache:
                self.cache[question_hash] = result.copy()
                self._save_cache()
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "sql": None,
                "source": "ai",
                "confidence": 0,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def fix_sql(self, sql: str, error: str) -> Dict[str, Any]:
        """
        Try to fix a SQL query that generated an error
        
        Args:
            sql: Original SQL query
            error: Error message from database
            
        Returns:
            Dictionary with fixed query details
        """
        start_time = time.time()
        
        try:
            # Prepare schema context if not already prepared
            self._prepare_schema_context()
            
            # Prepare params for validation chain
            params = {
                "schema": self.schema_context,
                "sql": sql,
                "error": error
            }
            
            # Add memory if enabled
            if self.use_memory:
                memory_context = self._prepare_memory_for_query(f"Fix SQL error: {error}") or ""
                params["memory"] = memory_context
            
            # Use validation chain to fix the SQL
            response = self.validation_chain.invoke(params)
            
            if isinstance(response, dict) and "text" in response:
                fixed_sql = response["text"].strip()
            else:
                return {
                    "success": False,
                    "sql": sql,
                    "fixed_sql": None,
                    "source": "ai",
                    "confidence": 0,
                    "error": "AI returned an unexpected response format",
                    "execution_time": time.time() - start_time
                }
            
            # Clean up response
            if fixed_sql.startswith('```') and fixed_sql.endswith('```'):
                fixed_sql = fixed_sql[3:-3].strip()
            if fixed_sql.startswith('sql') or fixed_sql.startswith('SQL'):
                fixed_sql = fixed_sql[3:].strip()
            
            # Validate the fixed SQL
            is_valid, validation_error = self._validate_sql(fixed_sql)
            if not is_valid:
                return {
                    "success": False,
                    "sql": sql,
                    "fixed_sql": fixed_sql,
                    "source": "ai",
                    "confidence": 40,
                    "error": validation_error,
                    "execution_time": time.time() - start_time
                }
            
            return {
                "success": True,
                "sql": sql,
                "fixed_sql": fixed_sql,
                "source": "ai",
                "confidence": 70,
                "execution_time": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "sql": sql,
                "fixed_sql": None,
                "source": "ai",
                "confidence": 0,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def execute_query(self, question: str, auto_fix: bool = True, max_attempts: int = 2) -> Dict[str, Any]:
        """
        Generate SQL from a natural language question and execute it
        
        Args:
            question: Natural language question
            auto_fix: Whether to automatically attempt to fix query errors
            max_attempts: Maximum number of auto-fix attempts
            
        Returns:
            Dictionary with execution results
        """
        # Generate the SQL
        generation_result = self.generate_sql(question)
        
        if not generation_result["success"] or not generation_result.get("sql"):
            return {
                "success": False,
                "question": question,
                "sql": None,
                "source": generation_result.get("source", "unknown"),
                "confidence": generation_result.get("confidence", 0),
                "error": generation_result.get("error", "Failed to generate SQL"),
                "execution_time": generation_result.get("execution_time", 0),
                "results": None
            }
        
        # Execute the SQL
        sql = generation_result["sql"]
        success, results, error = self.db_analyzer.execute_query(sql)
        
        # If execution failed and auto-fix is enabled, try to fix it
        attempts = 0
        while not success and auto_fix and attempts < max_attempts:
            attempts += 1
            fix_result = self.fix_sql(sql, error)
            
            if fix_result["success"] and fix_result.get("fixed_sql"):
                sql = fix_result["fixed_sql"]
                success, results, error = self.db_analyzer.execute_query(sql)
                
                # If the fix worked, update the query cache
                if success and self.use_cache:
                    question_hash = self._get_question_hash(question)
                    if question_hash in self.cache:
                        self.cache[question_hash]["sql"] = sql
                        self._save_cache()
            else:
                # If fixing failed, break the loop
                break
        
        # Store in memory if successful or even if failed (to remember errors too)
        if self.use_memory:
            self._store_in_memory(question, sql, results if success else None)
            
            # Update session context
            if success:
                self._update_session_context(question, sql, results)
            
        return {
            "success": success,
            "question": question,
            "sql": sql,
            "source": generation_result.get("source"),
            "confidence": generation_result.get("confidence"),
            "error": error,
            "auto_fixed": attempts > 0 and success,
            "fix_attempts": attempts,
            "execution_time": generation_result.get("execution_time", 0),
            "results": results
        }

    def _update_session_context(self, question: str, sql: str, results: List[Dict]) -> None:
        """
        Update session context with the latest query information
        
        Args:
            question: The natural language question
            sql: The generated SQL query
            results: The results of the query
        """
        if not self.use_memory:
            return
            
        # Extract personal info
        personal_info = self._extract_personal_info(question)
        if personal_info:
            for line in personal_info.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    self.session_context["user_info"][key.strip()] = value.strip()
        
        # Store query in sequence
        query_entry = {
            "question": question,
            "sql": sql,
            "timestamp": time.time(),
            "has_results": bool(results) and len(results) > 0
        }
        
        if results and len(results) > 0:
            # Store summary of results
            row_count = len(results)
            sample = results[0]
            columns = list(sample.keys()) if isinstance(sample, dict) else []
            
            query_entry["result_summary"] = {
                "row_count": row_count,
                "columns": columns,
                "sample": results[0] if row_count > 0 else None
            }
            
            # Extract important values from the results
            important_values = self._extract_important_values(question, sql, results)
            query_entry["important_values"] = important_values
            
            # Update the overall important values
            self.session_context["important_values"].update(important_values)
            
            # Store the last query result
            self.session_context["last_query_result"] = {
                "question": question,
                "sql": sql,
                "results": results[:10]  # Store up to 10 rows
            }
        
        # Add to query sequence
        self.session_context["query_sequence"].append(query_entry)
        if len(self.session_context["query_sequence"]) > 10:
            # Keep only the last 10 queries
            self.session_context["query_sequence"] = self.session_context["query_sequence"][-10:]
    
    def _extract_important_values(self, question: str, sql: str, results: List[Dict]) -> Dict[str, Any]:
        """
        Extract important values from query results that might be referenced in future queries
        
        Args:
            question: The natural language question
            sql: The SQL query that was executed
            results: The results of the query
            
        Returns:
            Dictionary of extracted important values
        """
        important_values = {}
        
        # Extract date values
        date_columns = set()
        if results and len(results) > 0:
            for col, value in results[0].items():
                # Check if the column might contain date values
                if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'month', 'year']):
                    date_columns.add(col)
                    
                    # Store the actual date values
                    if value is not None:
                        important_values[f"last_{col.lower().replace(' ', '_')}"] = value
                        
                        # For single-value results that are dates, also store special reference names
                        if len(results) == 1 and len(results[0]) <= 2:
                            col_lower = col.lower()
                            if 'first' in col_lower or 'min' in col_lower or 'earliest' in col_lower:
                                important_values["first_date"] = value
                                # Keep track of which question produced this value
                                important_values["first_date_question"] = question
                            elif 'last' in col_lower or 'max' in col_lower or 'latest' in col_lower:
                                important_values["last_date"] = value
                                # Keep track of which question produced this value
                                important_values["last_date_question"] = question
                            else:
                                important_values["the_date"] = value
        
        # Extract min/max values that might be referenced
        aggregation_prefixes = ['min', 'max', 'first', 'last', 'top', 'bottom', 'latest', 'earliest']
        for col, value in results[0].items() if results else []:
            col_lower = col.lower()
            if any(col_lower.startswith(prefix) for prefix in aggregation_prefixes):
                important_values[col_lower.replace(' ', '_')] = value
                
        # Check if the query is finding a specific entity
        common_queries = {
            r'SELECT.*MAX\((.*?)date\)': "last_date",
            r'SELECT.*MIN\((.*?)date\)': "first_date",
            r'SELECT.*TOP\s+1\s+.*ORDER\s+BY': "specific_entity",
            r'SELECT.*LIMIT\s+1': "specific_entity"
        }
        
        for pattern, value_type in common_queries.items():
            if re.search(pattern, sql, re.IGNORECASE) and results and len(results) == 1:
                # This is likely retrieving a specific value that will be referenced later
                for col, val in results[0].items():
                    if value_type == "last_date":
                        important_values["last_date"] = val
                        important_values["last_date_question"] = question
                    elif value_type == "first_date":
                        important_values["first_date"] = val
                        important_values["first_date_question"] = question
                    else:
                        # For other specific entities, store with column name
                        important_values[f"{value_type}_{col.lower().replace(' ', '_')}"] = val
        
        # Special handling for comparison queries - if we detect a request to compare,
        # Make sure we track both values being compared
        comparison_words = ["compare", "versus", "vs", "vs.", "which", "between", "difference", "more", "less", "higher", "lower", "greater", "better", "worse", "top", "bottom"]
        
        if any(word in question.lower() for word in comparison_words):
            important_values["is_comparison_query"] = True
            
            # If this is a follow-up to previous queries, link to those important values
            if "first_date" in self.session_context["important_values"] and "first_date" not in important_values:
                important_values["first_date"] = self.session_context["important_values"]["first_date"]
                if "first_date_question" in self.session_context["important_values"]:
                    important_values["first_date_question"] = self.session_context["important_values"]["first_date_question"]
                
            if "last_date" in self.session_context["important_values"] and "last_date" not in important_values:
                important_values["last_date"] = self.session_context["important_values"]["last_date"]
                if "last_date_question" in self.session_context["important_values"]:
                    important_values["last_date_question"] = self.session_context["important_values"]["last_date_question"]
        
        return important_values

    def _extract_sql_conditions(self, sql: str) -> str:
        """Extract WHERE, HAVING, JOIN conditions, and LIMIT/ORDER BY clauses from SQL"""
        conditions = []
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            conditions.append(f"WHERE {where_match.group(1).strip()}")
            
        # Extract HAVING clause
        having_match = re.search(r'HAVING\s+(.*?)(?:ORDER BY|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if having_match:
            conditions.append(f"HAVING {having_match.group(1).strip()}")
            
        # Extract JOIN conditions
        join_matches = re.finditer(r'(INNER|LEFT|RIGHT|FULL|OUTER)?\s*JOIN\s+(\S+)\s+(?:AS\s+\w+\s+)?ON\s+(.*?)(?=(?:LEFT|RIGHT|INNER|FULL|OUTER)?\s*JOIN|\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|$)', 
                                 sql, re.IGNORECASE | re.DOTALL)
        for match in join_matches:
            join_type = match.group(1) or "JOIN"
            table = match.group(2)
            join_condition = match.group(3).strip()
            conditions.append(f"{join_type} {table} ON {join_condition}")
        
        # Extract ORDER BY clause
        order_match = re.search(r'ORDER BY\s+(.*?)(?:LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if order_match:
            conditions.append(f"ORDER BY {order_match.group(1).strip()}")
            
        # Extract LIMIT clause
        limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
        if limit_match:
            conditions.append(f"LIMIT {limit_match.group(1).strip()}")
            
        return "; ".join(conditions)
        
    def _prepare_session_context_for_query(self, question: str) -> str:
        """
        Prepare relevant session context for the current query
        
        Args:
            question: The natural language question
            
        Returns:
            String with formatted session context
        """
        if not self.use_memory or not self.session_context:
            return ""
        
        context_parts = []
        
        # Add user information
        if self.session_context["user_info"]:
            user_info_str = "USER INFORMATION:\n"
            for key, value in self.session_context["user_info"].items():
                user_info_str += f"{key}: {value}\n"
            context_parts.append(user_info_str)
        
        # Check for references to previous queries
        reference_terms = [
            "this", "that", "these", "those", "it", "they", "them",
            "previous", "prior", "before", "last", "above", "mentioned",
            "same", "earlier"
        ]
        
        # Check for comparison terms
        comparison_terms = [
            "compare", "versus", "vs", "vs.", "which", "between", "difference", 
            "more", "less", "higher", "lower", "greater", "better", "worse",
            "top", "bottom", "each", "both", "two"
        ]
        
        # Check for superlative terms that might reference previous result set
        superlative_terms = [
            "highest", "lowest", "most", "least", "best", "worst", "largest", "smallest",
            "maximum", "minimum", "max", "min"
        ]
        
        # Check if query contains references to "those" or other terms that imply previous result set
        has_result_set_reference = any(f"{term} " in f" {question.lower()} " for term in ["those", "these", "them"])
        
        # Check for implicit references or refinement patterns
        refinement_starters = [
            "just", "only", "show", "list", "give", "select", "filter",
            "now", "then", "also", "and", "but"
        ]
        
        implicit_words = ["column", "row", "field", "value", "record", "result", "data"]
        
        # Determine if this looks like a refinement or follow-up query
        is_refinement = False
        is_reference = any(term in question.lower() for term in reference_terms)
        is_comparison = any(term in question.lower() for term in comparison_terms)
        is_superlative = any(term in question.lower() for term in superlative_terms)
        
        # Check if it starts with a refinement word
        if any(question.lower().startswith(term) for term in refinement_starters):
            is_refinement = True
            
        # Check for implicit reference patterns (like "just the ids and the date column")
        if any(word in question.lower() for word in implicit_words):
            is_refinement = True
            
        # Check for very short queries that are likely refinements
        if len(question.split()) <= 7:
            is_refinement = True
        
        # Always include the last query if it exists
        if self.session_context["last_query_result"]:
            last_result = self.session_context["last_query_result"]
            
            last_query_str = "PREVIOUS QUERY:\n"
            last_query_str += f"Question: {last_result['question']}\n"
            last_query_str += f"SQL: {last_result['sql']}\n"
            
            # Extract the conditions from the last SQL query if this looks like a refinement
            if is_refinement:
                # Extract WHERE/HAVING/JOIN conditions
                last_sql = last_result['sql']
                conditions = self._extract_sql_conditions(last_sql)
                if conditions:
                    last_query_str += f"Conditions: {conditions}\n"
                    
                # Extract the table names for context
                tables = self._extract_sql_tables(last_sql)
                if tables:
                    last_query_str += f"Tables: {', '.join(tables)}\n"
            
            # Include sample results
            if last_result["results"]:
                row_count = len(last_result["results"])
                if row_count == 1:
                    # For single row results, show the actual values clearly
                    last_query_str += "Result (single row):\n"
                    for col, val in last_result["results"][0].items():
                        last_query_str += f"- {col}: {val}\n"
                else:
                    # For multiple rows, summarize
                    last_query_str += f"Results: {row_count} rows with columns: "
                    if last_result["results"][0]:
                        last_query_str += ", ".join(last_result["results"][0].keys())
                    last_query_str += "\n"
                    
                    # If there's a reference to "those" or similar terms AND the previous query used LIMIT
                    if has_result_set_reference and "LIMIT" in last_result['sql'].upper():
                        last_query_str += "\nIMPORTANT: When referring to 'those' or 'these' results, you MUST preserve the exact same result set from the previous query.\n"
                        
                        # Extract IDs if available
                        if last_result["results"] and "salesorderid" in last_result["results"][0]:
                            ids = [str(row["salesorderid"]) for row in last_result["results"]]
                            last_query_str += f"Previous result set contains ONLY these IDs: {', '.join(ids)}\n"
                            last_query_str += "Use these exact IDs in a WHERE clause: WHERE salesorderid IN (" + ', '.join(ids) + ")\n"
            
            context_parts.append(last_query_str)
        
        # For refinements or references, add more context about previous queries
        if is_refinement or is_reference or is_superlative:
            # Add specific guidance for refinements
            guidance = "GUIDANCE FOR REFINEMENT:\n"
            
            if has_result_set_reference and is_superlative:
                guidance += "CRITICAL: This query refers to the SPECIFIC SET of results from the previous query. "
                guidance += "Do NOT run a new query against the entire table. "
                guidance += "You MUST restrict the query to ONLY the exact rows returned by the previous query.\n"
            elif is_refinement and not is_reference:
                guidance += "This appears to be a refinement of the previous query. "
                guidance += "Maintain the same filters and conditions from the previous query, "
                guidance += "but modify the output columns or presentation according to the new request.\n"
            context_parts.append(guidance)
            
            # Include important values
            if self.session_context["important_values"]:
                values_str = "IMPORTANT VALUES FROM PREVIOUS QUERIES:\n"
                for key, value in self.session_context["important_values"].items():
                    # Skip metadata keys
                    if not key.endswith('_question') and not key.startswith('is_'):
                        values_str += f"{key}: {value}\n"
                context_parts.append(values_str)
        
            # Include more context from earlier queries
            if len(self.session_context["query_sequence"]) > 1:
                earlier_queries = self.session_context["query_sequence"][:-1][-2:]  # Get up to 2 earlier queries
                
                earlier_str = "EARLIER QUERIES:\n"
                for idx, query in enumerate(earlier_queries):
                    earlier_str += f"Query {len(earlier_queries) - idx}:\n"
                    earlier_str += f"- Question: {query['question']}\n"
                    earlier_str += f"- SQL: {query['sql']}\n"
                    
                    if query.get("important_values"):
                        important_vals = {k: v for k, v in query["important_values"].items() 
                                         if not k.endswith('_question') and not k.startswith('is_')}
                        if important_vals:
                            earlier_str += "- Important values: "
                            earlier_str += ", ".join([f"{k}: {v}" for k, v in important_vals.items()])
                            earlier_str += "\n"
                
                context_parts.append(earlier_str)
                
        # Special handling for comparison queries
        if is_comparison:
            comparison_context = "COMPARISON CONTEXT:\n"
            
            # Check if we have cached first_date and last_date
            first_date = self.session_context["important_values"].get("first_date")
            last_date = self.session_context["important_values"].get("last_date")
            
            if first_date and last_date:
                comparison_context += f"You are being asked to compare values between these dates:\n"
                
                first_date_q = self.session_context["important_values"].get("first_date_question", "")
                if first_date_q:
                    comparison_context += f"- First date ({first_date}) was from query: \"{first_date_q}\"\n"
                else:
                    comparison_context += f"- First date: {first_date}\n"
                    
                last_date_q = self.session_context["important_values"].get("last_date_question", "")
                if last_date_q:
                    comparison_context += f"- Last date ({last_date}) was from query: \"{last_date_q}\"\n"
                else:
                    comparison_context += f"- Last date: {last_date}\n"
                    
                comparison_context += "Make sure to include BOTH dates in your comparison query.\n"
                context_parts.append(comparison_context)
            
        return "\n".join(context_parts)
        
    def _extract_sql_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL query"""
        tables = []
        
        # Extract FROM clause
        from_match = re.search(r'FROM\s+(.*?)(?:WHERE|GROUP BY|ORDER BY|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if from_match:
            # Split by commas and clean up
            from_tables = from_match.group(1).strip().split(',')
            for table in from_tables:
                # Remove aliases and schema prefixes for simplicity
                table_clean = re.sub(r'(?:AS)?\s+\w+\s*$', '', table.strip(), flags=re.IGNORECASE)
                tables.append(table_clean.strip())
                
        # Extract JOIN tables
        join_matches = re.finditer(r'JOIN\s+(\S+)', sql, re.IGNORECASE)
        for match in join_matches:
            tables.append(match.group(1).strip())
            
        return tables

    def _prepare_memory_for_query(self, question: str) -> str:
        """Prepare memory context specifically for the current query"""
        memory_context = ""
        
        # First get session context (in-memory)
        session_context = self._prepare_session_context_for_query(question)
        if session_context:
            memory_context += session_context + "\n\n"
        
        # Then add vector store memory if available
        if self.memory and self.use_memory:
            try:
                # Get relevant memories
                vector_memory = self._get_memory_context(question)
                
                if vector_memory:
                    # Extract user information patterns
                    user_info = []
                    for line in vector_memory.split('\n'):
                        if line.startswith('User '):
                            user_info.append(line)
                    
                    # Add vector memory without duplicating user info already in session context
                    if session_context and user_info:
                        # Remove the user info from vector memory to avoid duplication
                        memory_lines = [line for line in vector_memory.split('\n') if not line.startswith('User ')]
                        filtered_memory = "\n".join(memory_lines)
                        memory_context += "PERSISTENT MEMORY:\n" + filtered_memory
                    else:
                        memory_context += vector_memory
            except Exception as e:
                print(f"Error preparing vector memory for query: {e}")
        
        return memory_context


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Database connection parameters
    db_name = os.getenv("DB_NAME", "postgres")
    username = os.getenv("DB_USERNAME", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    
    # Initialize database analyzer
    db_analyzer = DatabaseAnalyzer(db_name, username, password, host, port)
    
    # Initialize SQL generator
    sql_generator = SmartSQLGenerator(
        db_analyzer,
        use_memory=True,
        memory_persist_dir="./memory_store"
    )
    
    # Test with a sample question
    question = "Show me the top 5 customers by total order amount"
    result = sql_generator.execute_query(question)
    
    if result["success"]:
        print(f"Question: {question}")
        print(f"SQL: {result['sql']}")
        print(f"Results: {len(result['results'])} rows")
        
        # Display first few results
        if result["results"]:
            import pandas as pd
            df = pd.DataFrame(result["results"])
            print("\nSample results:")
            print(df.head()) 