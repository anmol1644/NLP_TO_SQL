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
        cache_file: str = "query_cache.json"
    ):
        """
        Initialize the SQL generator
        
        Args:
            db_analyzer: Database analyzer instance
            model_name: Generative AI model to use
            use_cache: Whether to cache query results
            cache_file: Path to the query cache file
        """
        load_dotenv()
        self.db_analyzer = db_analyzer
        self.model_name = model_name
        self.use_cache = use_cache
        self.cache_file = cache_file
        self.cache = self._load_cache() if use_cache else {}
        
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
        
        # Define SQL generation prompt
        self.sql_prompt = PromptTemplate(
            input_variables=["schema", "question", "examples"],
            template="""You are an expert SQL developer specializing in PostgreSQL databases. Your job is to translate natural language questions into precise and efficient SQL queries.

### DATABASE SCHEMA:
{schema}

### EXAMPLES OF GOOD SQL PATTERNS:
{examples}

### TASK:
Convert the following question into a single PostgreSQL SQL query:
"{question}"

### GUIDELINES:
1. Create only PostgreSQL-compatible SQL
2. Focus on writing efficient queries
3. Use proper table aliases for clarity
4. Include appropriate JOINs based on database relationships
5. Include comments explaining complex parts of your query
6. Always use proper column quoting with double quotes for columns with spaces or special characters
7. NEVER use any placeholder values in your final query

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
            input_variables=["schema", "sql", "error"],
            template="""You are an expert SQL developer specializing in PostgreSQL databases. Your job is to fix SQL query errors.

### DATABASE SCHEMA:
{schema}

### QUERY WITH ERROR:
```sql
{sql}
```

### ERROR MESSAGE:
{error}

### TASK:
Fix the SQL query to resolve the error.

### GUIDELINES:
1. Create only PostgreSQL-compatible SQL
2. Maintain the original query intent
3. Fix any syntax errors, typos, or invalid column references
4. NEVER use any placeholder values in your final query

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
            # Generate SQL with LangChain + Gemini
            response = self.sql_chain.invoke({
                "schema": self.schema_context,
                "question": question,
                "examples": self.example_patterns
            })
            
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
            
            # Use validation chain to fix the SQL
            response = self.validation_chain.invoke({
                "schema": self.schema_context,
                "sql": sql,
                "error": error
            })
            
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
    sql_generator = SmartSQLGenerator(db_analyzer)
    
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