import os
import time
import hashlib
import json
import re
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union

import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableSequence
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.schema.memory import BaseMemory
from langchain.schema import BaseRetriever

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
        
        # Data store for paginated results
        self.paginated_results = {}
        
        # Session-specific memory for tracking conversation context
        self.session_context = {
            "user_info": {},
            "query_sequence": [],
            "important_values": {},
            "last_query_result": None,
            "entity_mentions": {},
            "text_responses": [],  # Store text responses
            "multi_query_results": []  # Store results from multiple queries
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
16. IMPORTANT: When the user asks for "all" or "list all" data, DO NOT use aggregation functions (SUM, COUNT, AVG) unless explicitly requested. Return the raw data rows.
17. When the user asks to "show" or "list" data without explicitly asking for aggregation, return the individual rows rather than summary statistics.

### OUTPUT FORMAT:
Provide ONLY the SQL query with no additional text, explanation, or markdown formatting.
"""
        )
        
        # Create the SQL generation chain using RunnableSequence
        self.sql_chain = self.sql_prompt | self.llm
        
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
        
        # Create the validation chain using RunnableSequence
        self.validation_chain = self.validation_prompt | self.llm

        # Define text response prompt
        self.text_response_prompt = PromptTemplate(
            input_variables=["schema", "question", "results", "sql"] + (["memory"] if use_memory else []),
            template=f"""You are a helpful database assistant who helps answer questions about data.

{memory_var}### DATABASE SCHEMA:
{{schema}}

### USER QUESTION:
"{{question}}"

### SQL QUERY:
```sql
{{sql}}
```

### QUERY RESULTS:
```
{{results}}
```

### TASK:
Based on the question and the SQL query results, provide a natural language response.

### GUIDELINES:
1. If the user asked for specific data analysis, provide insights and analysis based on the results
2. If the user asked a question that doesn't require SQL, answer it directly based on your knowledge
3. Keep your response concise and focused on answering the question
4. Include relevant numbers and metrics from the results if available
5. Format numeric values appropriately (e.g., large numbers with commas, percentages, currencies)
6. For small result sets, you can mention specific data points
7. For large result sets, summarize the overall trends or patterns
8. If the results are empty, explain what that means in context
9. Use the schema information to provide more context when needed
10. If there was an error in the SQL query, explain what might have gone wrong
11. For questions about the system itself:
   - If asked about what LLM you're using, say you're using Google's Gemini model
   - If asked about the system architecture, explain it's a natural language to SQL system using LLMs
   - If asked about capabilities, explain you can translate natural language to SQL and analyze data
   - Be honest and straightforward about your capabilities and limitations

### OUTPUT FORMAT:
Provide a natural language response that directly answers the user's question. Be helpful, clear, and concise.
"""
        )
        
        # Create the text response chain using RunnableSequence
        self.text_response_chain = self.text_response_prompt | self.llm

        # Define conversation prompt for non-SQL conversations
        self.conversation_prompt = PromptTemplate(
            input_variables=["schema", "question"] + (["memory"] if use_memory else []),
            template=f"""You are a helpful database assistant who helps answer questions about data and databases.

{memory_var}### DATABASE SCHEMA:
{{schema}}

### USER QUESTION:
"{{question}}"

### TASK:
Respond to the user's question. This appears to be a general question that doesn't require generating SQL.

### GUIDELINES:
1. If the question is about databases or data concepts, provide a helpful explanation
2. If the question is about the schema or structure of the database, refer to the schema information
3. If the question is unrelated to databases, provide a general helpful response
4. Be concise and direct in your response
5. Use your knowledge about databases and SQL when relevant
6. If the question might benefit from executing SQL but is currently phrased as a conversation, suggest what specific data the user might want to query
7. For questions about the system itself:
   - If asked about what LLM you're using, say you're using Google's Gemini model
   - If asked about the system architecture, explain it's a natural language to SQL system using LLMs
   - If asked about capabilities, explain you can translate natural language to SQL and analyze data
   - Be honest and straightforward about your capabilities and limitations

### OUTPUT FORMAT:
Provide a natural language response that directly answers the user's question. Be helpful, clear, and concise.
"""
        )
        
        # Create the conversation chain using RunnableSequence
        self.conversation_chain = self.conversation_prompt | self.llm

        # Define multi-query analysis prompt
        self.analysis_prompt = PromptTemplate(
            input_variables=["schema", "question", "tables_info"] + (["memory"] if use_memory else []),
            template=f"""You are an expert data analyst who helps analyze database query results.

{memory_var}### DATABASE SCHEMA:
{{schema}}

### USER QUESTION:
"{{question}}"

### QUERY RESULTS:
{{tables_info}}

### TASK:
Analyze the query results to answer the user's question. The user has requested a complex analysis that required multiple queries.

### GUIDELINES:
1. Compare and analyze data across the multiple tables provided
2. Identify trends, anomalies, or patterns in the data
3. Provide insights that directly answer the user's question
4. Reference specific numbers from the results to support your analysis
5. Use proper statistical reasoning when making comparisons
6. For time-based comparisons (such as year-over-year), calculate and explain percentage changes
7. Be concise yet thorough in your explanation
8. Format numbers appropriately (with commas for thousands, percentages with % sign)
9. If appropriate, suggest potential reasons for patterns observed in the data
10. When analyzing financial data, consider both absolute and relative changes
11. For comparisons between periods, highlight significant changes and potential causes

### OUTPUT FORMAT:
Provide a thorough analysis of the data that directly answers the user's question. Be insightful, clear, and data-driven.
"""
        )
        
        # Create the analysis chain using RunnableSequence
        self.analysis_chain = self.analysis_prompt | self.llm

        # Define query planning prompt for complex questions
        self.query_planner_prompt = PromptTemplate(
            input_variables=["schema", "question"] + (["memory"] if use_memory else []),
            template=f"""You are an expert SQL database analyst with deep knowledge of PostgreSQL databases. Your job is to plan the SQL queries needed to answer complex questions.

{memory_var}### DATABASE SCHEMA:
{{schema}}

### USER QUESTION:
"{{question}}"

### TASK:
Identify if the user's question requires a single SQL query or multiple SQL queries for proper analysis. 
For simple data retrieval, a single query is sufficient. For questions involving comparison, trend analysis, or "explain why" type questions, multiple queries are typically needed.

### GUIDELINES:
1. Analyze if the question requires multiple SQL queries or just a single query
2. For comparison questions (e.g., "compare to last year", "why are sales down"), plan multiple queries
3. Identify specific time periods, metrics, or entities that need to be queried separately

### OUTPUT FORMAT:
Answer in key-value format, following this exact format:

NEEDS_MULTIPLE_QUERIES: yes/no
NUMBER_OF_QUERIES_NEEDED: [number]
QUERY_PLAN:
- Query 1: [description of first query]
- Query 2: [description of second query]
...
ANALYSIS_APPROACH: [brief description of how these queries will help answer the question]
"""
        )
        
        # Create the query planner chain using RunnableSequence
        self.query_planner_chain = self.query_planner_prompt | self.llm
        
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
            
            # Create custom memory class
            class CustomVectorMemory(BaseMemory):
                def __init__(self, retriever):
                    self.retriever = retriever
                    self.memory_key = "memory"
                    
                @property
                def memory_variables(self) -> List[str]:
                    return [self.memory_key]
                
                def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                    query = inputs.get("question", "")
                    if query:
                        try:
                            docs = self.retriever.get_relevant_documents(query)
                            memory_content = "\n".join([doc.page_content for doc in docs])
                            return {self.memory_key: memory_content}
                        except Exception as e:
                            print(f"Error loading memory: {e}")
                            return {self.memory_key: ""}
                    return {self.memory_key: ""}
                
                def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
                    try:
                        content = outputs.get("memory", "")
                        if content:
                            doc = Document(page_content=content)
                            self.retriever.vectorstore.add_documents([doc])
                    except Exception as e:
                        print(f"Error saving to memory: {e}")
                
                def clear(self) -> None:
                    # Clear method - not implemented for vector store
                    pass
            
            memory = CustomVectorMemory(retriever)
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

    def _extract_response_content(self, response) -> str:
        """Extract content from LangChain response in a consistent way"""
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict) and "text" in response:
            return response["text"]
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def _get_memory_context(self, question: str) -> str:
        """Retrieve relevant context from memory for a question"""
        if not self.memory or not self.use_memory:
            return ""
            
        try:
            # Get relevant memories
            memory_context = self.memory.load_memory_variables({"question": question})
            memory_data = memory_context.get("memory", "")
            
            # Ensure we return a string
            if isinstance(memory_data, list):
                # If it's a list of documents, join their content
                memory_strings = []
                for doc in memory_data:
                    if hasattr(doc, 'page_content'):
                        memory_strings.append(doc.page_content)
                    elif isinstance(doc, str):
                        memory_strings.append(doc)
                    else:
                        memory_strings.append(str(doc))
                return "\n".join(memory_strings)
            elif isinstance(memory_data, str):
                return memory_data
            else:
                return str(memory_data)
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
            
            sql_response = self._extract_response_content(response)
            
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
            
            fixed_sql = self._extract_response_content(response).strip()
            
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
    
    def generate_text_response(self, question: str, sql: str = None, results: Any = None) -> Dict[str, Any]:
        """
        Generate a natural language text response based on the question, SQL query, and results
        
        Args:
            question: Natural language question
            sql: SQL query (if available)
            results: Query results (if available)
            
        Returns:
            Dictionary with text generation details
        """
        start_time = time.time()
        
        # Prepare schema context if not already prepared
        self._prepare_schema_context()
        
        try:
            # Determine if this is a conversational query or needs SQL analysis
            is_sql_related = sql is not None and results is not None
            
            if is_sql_related:
                # Format results for the model
                results_formatted = self._format_results_for_display(results)
                
                # Prepare params for text response generation
                params = {
                    "schema": self.schema_context,
                    "question": question,
                    "sql": sql,
                    "results": results_formatted
                }
                
                # Add memory if enabled
                if self.use_memory:
                    memory_context = self._prepare_memory_for_query(question) or ""
                    params["memory"] = memory_context
                    
                # Generate text response with SQL analysis
                response = self.text_response_chain.invoke(params)
            else:
                # This is a conversational query, no SQL needed
                params = {
                    "schema": self.schema_context,
                    "question": question
                }
                
                # Add memory if enabled
                if self.use_memory:
                    memory_context = self._prepare_memory_for_query(question) or ""
                    params["memory"] = memory_context
                
                # Generate conversational response
                response = self.conversation_chain.invoke(params)
            
            text_response = self._extract_response_content(response).strip()
            
            # Store in session context for future reference
            if self.session_context:
                self.session_context["text_responses"].append({
                    "question": question,
                    "text": text_response,
                    "timestamp": time.time()
                })
                
                # Limit stored responses to last 10
                if len(self.session_context["text_responses"]) > 10:
                    self.session_context["text_responses"] = self.session_context["text_responses"][-10:]
            
            # Store in memory if enabled
            if self.use_memory:
                self._store_text_in_memory(question, text_response, sql, results)
            
            return {
                "success": True,
                "text": text_response,
                "generation_time": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "text": f"Error generating text response: {str(e)}",
                "generation_time": time.time() - start_time
            }
    
    def _format_results_for_display(self, results: Any) -> str:
        """Format query results for display in text responses"""
        if not results:
            return "No results returned."
        
        try:
            # For list results, format as a table or list
            if isinstance(results, list):
                if not results:
                    return "Empty result set."
                
                # Convert to pandas DataFrame for string formatting
                import pandas as pd
                df = pd.DataFrame(results)
                
                # Format with reasonable width constraints
                with pd.option_context('display.max_rows', 20, 'display.max_columns', 20, 'display.width', 1000):
                    formatted = df.to_string(index=False)
                    
                    # If too long, truncate
                    if len(formatted) > 4000:
                        formatted = formatted[:4000] + "\n... [truncated]"
                    
                    return formatted
            else:
                # For other types, use string representation
                return str(results)
        except Exception as e:
            return f"Error formatting results: {e}"
    
    def _store_text_in_memory(self, question: str, text_response: str, sql: str = None, results: Any = None) -> None:
        """Store text responses in memory for future reference"""
        if not self.memory or not self.use_memory:
            return
            
        try:
            # Create content with question and response
            content = f"Question: {question}\nResponse: {text_response}"
            
            # Add SQL context if available
            if sql:
                content += f"\nSQL: {sql}"
            
            # Add result summary if available (brief)
            if results and isinstance(results, list):
                num_rows = len(results)
                content += f"\nReturned {num_rows} rows"
            
            # Store in memory
            self.memory.save_context(
                {"question": question}, 
                {"memory": content}
            )
        except Exception as e:
            print(f"Error storing text in memory: {e}")

    def _is_analysis_question(self, question: str) -> bool:
        """
        Check if the question requires data analysis with potentially multiple queries
        
        Args:
            question: The natural language question
            
        Returns:
            True if the question requires analysis with potentially multiple queries
        """
        # Keywords indicating analysis
        analysis_keywords = [
            "analyze", "analysis", "compare", "comparison", "trend", 
            "why", "reason", "explain", "growth", "decline", "difference",
            "versus", "vs", "against", "performance", "over time",
            "year over year", "month over month", "quarter over quarter",
            "decreasing", "increasing", "dropping", "rising", "falling", 
            "higher", "lower", "better", "worse", "improved", "deteriorated"
        ]
        
        # Check for comparison between time periods
        time_comparisons = [
            r"compare.*last\s+(year|month|quarter|week|period)",
            r"compare.*previous\s+(year|month|quarter|week|period)",
            r"(higher|lower|more|less|greater|fewer).*than\s+(last|previous)\s+(year|month|quarter|week|period)",
            r"(increase|decrease|change|drop|rise|fall)\s+from\s+(last|previous)\s+(year|month|quarter|week|period)",
            r"(increase|decrease|change|drop|rise|fall)\s+since\s+(last|previous)\s+(year|month|quarter|week|period)",
            r"(increase|decrease|change|drop|rise|fall)\s+compared to\s+(last|previous)\s+(year|month|quarter|week|period)",
            r"(this|current)\s+(year|month|quarter|week|period).*compared to",
            r"(this|current).*vs\.?\s+(last|previous)",
            r"why\s+(is|are|were|was|have|has|had)\s+.*(increase|decrease|change|higher|lower|more|less|drop|rise|fall)"
        ]
        
        # Check if the question starts with "why" - these almost always need analysis
        if question.lower().strip().startswith("why"):
            return True
            
        # Check if the question contains analysis keywords
        question_lower = question.lower()
        
        # Strong indicators of needing analysis
        if any(keyword in question_lower for keyword in analysis_keywords):
            return True
            
        # Check for time comparison patterns
        for pattern in time_comparisons:
            if re.search(pattern, question_lower):
                return True
                
        return False
    
    def _is_why_question(self, question: str) -> bool:
        """
        Check if the question is a 'why' question that requires deeper causal analysis
        
        Args:
            question: The natural language question
            
        Returns:
            True if this is a 'why' question requiring causal analysis
        """
        question_lower = question.lower().strip()
        
        # Direct "why" questions
        if question_lower.startswith("why"):
            return True
            
        # Questions about reasons or causes
        reason_patterns = [
            r"what\s+(is|are|were|was)\s+the\s+reason",
            r"what\s+caused",
            r"reason\s+for",
            r"cause\s+of",
            r"explain\s+why",
            r"how\s+come",
            r"tell\s+me\s+why",
            r"factors\s+(behind|causing)",
            r"what\s+explains",
        ]
        
        for pattern in reason_patterns:
            if re.search(pattern, question_lower):
                return True
                
        return False
        
    def handle_why_question(self, question: str) -> Dict[str, Any]:
        """
        Handle 'why' questions that require deeper causal analysis between time periods
        
        Args:
            question: The natural language question asking for causal analysis
            
        Returns:
            Dictionary with execution results including comparative analysis
        """
        start_time = time.time()
        
        print(f"Handling 'why' question: {question}")
        
        # Step 1: Determine the key metrics and time periods that need to be compared
        time_periods_to_analyze = self._extract_time_periods_from_question(question)
        
        if not time_periods_to_analyze:
            # Default to comparing current period with previous period
            time_periods_to_analyze = [
                {"name": "Current period", "description": "Most recent data"},
                {"name": "Previous period", "description": "Period before the most recent data for comparison"}
            ]
        
        # Step 2: Generate queries for each time period
        query_results = []
        tables_info = []
        
        for period in time_periods_to_analyze:
            # Construct a query specific to this time period
            period_question = f"Show the relevant metrics for {period['name']}: {period['description']}"
            
            # Add the original question for context
            period_question += f". This is part of answering: {question}"
            
            # Generate SQL for this specific time period
            sql_generation = self.generate_sql(period_question)
            
            if not sql_generation["success"] or not sql_generation.get("sql"):
                print(f"Failed to generate SQL for {period['name']}")
                continue
                
            sql = sql_generation["sql"]
            
            # Execute the SQL
            success, results, error = self.db_analyzer.execute_query(sql)
            
            if not success or not results:
                print(f"Failed to get results for {period['name']}: {error}")
                continue
                
            # Store the successful query and results
            query_results.append({
                "query_name": period["name"],
                "sql": sql,
                "results": results,
                "row_count": len(results),
                "description": period["description"]
            })
        
        # If we don't have at least two successful queries for comparison, try a combined approach
        if len(query_results) < 2:
            print("Insufficient period data, attempting combined query approach")
            
            # Generate a single comprehensive query that includes time period as a dimension
            combined_question = f"Show data across different time periods to analyze {question}"
            sql_generation = self.generate_sql(combined_question)
            
            if sql_generation["success"] and sql_generation.get("sql"):
                sql = sql_generation["sql"]
                success, results, error = self.db_analyzer.execute_query(sql)
                
                if success and results:
                    query_results.append({
                        "query_name": "Combined period analysis",
                        "sql": sql,
                        "results": results,
                        "row_count": len(results),
                        "description": "Combined analysis across time periods"
                    })
        
        # If still no successful queries, return error
        if not query_results:
            print("No successful queries for 'why' analysis")
            return {
                "success": False,
                "question": question,
                "error": "Failed to retrieve data for analysis",
                "execution_time": time.time() - start_time
            }
        
        # Step 3: Process results for analysis
        print(f"Preparing {len(query_results)} query results for 'why' analysis")
        for qr in query_results:
            results = qr["results"]
            query_name = qr["query_name"]
            
            # Apply the same large result handling as in execute_multi_query_analysis
            if len(results) > 100:
                # Include statistics for large result sets
                import pandas as pd
                import numpy as np
                
                df = pd.DataFrame(results)
                
                # Get statistics
                stats = {}
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    stats[col] = {
                        "min": df[col].min() if not pd.isna(df[col].min()) else "N/A",
                        "max": df[col].max() if not pd.isna(df[col].max()) else "N/A",
                        "mean": df[col].mean() if not pd.isna(df[col].mean()) else "N/A",
                        "median": df[col].median() if not pd.isna(df[col].median()) else "N/A"
                    }
                
                # Sample rows
                sampled_rows = []
                sampled_rows.extend(results[:20])
                
                if len(results) > 40:
                    sampled_rows.append({"note": "... skipping middle rows ..."})
                    sampled_rows.extend(results[-20:])
                
                table_formatted = (
                    f"LARGE RESULT SET: {len(results)} rows total, showing first 20 and last 20 rows.\n\n"
                    f"STATISTICS FOR NUMERIC COLUMNS:\n{json.dumps(stats, indent=2)}\n\n"
                    f"SAMPLED ROWS:\n{self._format_results_for_display(sampled_rows)}"
                )
            else:
                table_formatted = self._format_results_for_display(results)
            
            # Add formatted table to tables_info
            tables_info.append(
                f"### {query_name} ###\n"
                f"DESCRIPTION: {qr.get('description', '')}\n"
                f"SQL:\n```sql\n{qr['sql']}\n```\n\n"
                f"RESULTS ({qr['row_count']} rows):\n{table_formatted}\n"
            )
        
        # Step 4: Generate specialized causal analysis with focus on explaining "why"
        print("Generating causal analysis for 'why' question")
        tables_info_text = "\n\n".join(tables_info)
        
        # Create specialized prompt that focuses on explaining causes
        causal_analysis_prompt = PromptTemplate(
            input_variables=["schema", "question", "tables_info"] + (["memory"] if self.use_memory else []),
            template=f"""You are an expert data analyst specialized in explaining causes and reasons behind business trends.

{"{{memory}}\n\n" if self.use_memory else ""}### DATABASE SCHEMA:
{{schema}}

### USER QUESTION:
"{{question}}"

### QUERY RESULTS:
{{tables_info}}

### TASK:
Provide a detailed causal analysis answering the user's "why" question. Focus on explaining the reasons behind the observed trend or phenomenon.

### GUIDELINES:
1. Identify the key metrics that have changed between time periods
2. Calculate the percentage changes for important metrics
3. Look for patterns or anomalies that could explain the changes
4. Consider multiple potential causes for the observed patterns
5. Analyze both direct and indirect factors that might contribute
6. Compare metrics across different dimensions (time, products, regions, etc.)
7. Explain which factors appear most significant based on the data
8. Be specific about the magnitude and direction of changes
9. Use concrete numbers and percentages from the data to support your explanation
10. Rank the likely causes in order of impact when possible
11. Acknowledge limitations in the analysis if the data doesn't fully explain the trend

### OUTPUT FORMAT:
Provide a thorough analysis that directly answers why the observed trend is happening. Structure your response with clear sections covering different potential causes.
"""
        )
        
        # Create a temporary chain for the causal analysis
        causal_chain = causal_analysis_prompt | self.llm
        
        # Prepare params for analysis
        params = {
            "schema": self.schema_context,
            "question": question,
            "tables_info": tables_info_text
        }
        
        # Add memory if enabled
        if self.use_memory:
            memory_context = self._prepare_memory_for_query(question) or ""
            params["memory"] = memory_context
        
        # Generate the causal analysis
        analysis_response = causal_chain.invoke(params)
        
        analysis_text = self._extract_response_content(analysis_response).strip()
        print("Causal analysis generated successfully")
        
        # Step 5: Store in memory and session context
        if self.use_memory:
            self._store_text_in_memory(question, analysis_text)
            
            for qr in query_results:
                self._store_in_memory(
                    f"Data for why analysis: {question} - {qr['query_name']}", 
                    qr['sql'], 
                    qr['results']
                )
        
        # Store in session context for future reference
        self.session_context["multi_query_results"] = query_results
        self.session_context["text_responses"].append({
            "question": question,
            "text": analysis_text,
            "timestamp": time.time()
        })
        
        print(f"'Why' analysis completed in {time.time() - start_time:.2f} seconds")
        
        # Step 6: Return the full result with all tables
        return {
            "success": True,
            "question": question,
            "is_multi_query": True,
            "is_why_analysis": True,
            "tables": query_results,
            "text": analysis_text,
            "execution_time": time.time() - start_time
        }
    
    def _extract_time_periods_from_question(self, question: str) -> List[Dict[str, str]]:
        """
        Extract time periods to analyze from the question
        
        Args:
            question: The natural language question
            
        Returns:
            List of time periods to analyze with name and description
        """
        question_lower = question.lower()
        time_periods = []
        
        # Common time period patterns
        period_patterns = {
            "current_year": r"(this|current|present)\s+year",
            "last_year": r"last\s+year|previous\s+year|year\s+ago",
            "current_quarter": r"(this|current|present)\s+quarter",
            "last_quarter": r"last\s+quarter|previous\s+quarter|quarter\s+ago",
            "current_month": r"(this|current|present)\s+month",
            "last_month": r"last\s+month|previous\s+month|month\s+ago"
        }
        
        for period_id, pattern in period_patterns.items():
            if re.search(pattern, question_lower):
                if "current" in period_id or "this" in period_id:
                    name = period_id.replace("_", " ").replace("current", "current").title()
                    time_periods.append({
                        "name": name,
                        "description": f"Data for {name}"
                    })
                elif "last" in period_id or "previous" in period_id:
                    name = period_id.replace("_", " ").replace("last", "previous").title()
                    time_periods.append({
                        "name": name,
                        "description": f"Data for {name}"
                    })
        
        # If we found specific periods, return them
        if time_periods:
            return time_periods
            
        # Default time periods if none were explicitly mentioned
        return [
            {"name": "Current Period", "description": "Most recent data"},
            {"name": "Previous Period", "description": "Period before the most recent data"}
        ]
    
    def _is_sql_question(self, question: str) -> bool:
        """
        Determine if a question requires SQL generation or is conversational
        
        Args:
            question: The natural language question
            
        Returns:
            True if the question likely requires SQL, False if it's conversational
        """
        # Keywords indicating SQL is needed
        sql_keywords = [
            "show me", "list", "find", "query", "select", "data", "database",
            "table", "record", "rows", "search", "get", "fetch", "retrieve",
            "count", "sum", "average", "total", "calculate", "analyze", "report",
            "compare", "filter", "sort", "order", "group", "join", "where",
            "how many", "which", "when", "sales", "customer", "order", "product"
        ]
        
        # Keywords indicating conversation
        conversation_keywords = [
            "hello", "hi ", "hey", "thanks", "thank you", "help", "explain",
            "what is", "how do", "why", "can you", "please", "would", "could",
            "definition", "mean", "define"
        ]
        
        # Question about the database structure
        schema_keywords = [
            "schema", "structure", "tables", "columns", "relationships", 
            "foreign keys", "primary keys"
        ]
        
        # Check for SQL-like patterns
        question_lower = question.lower()
        
        # Direct check for schema queries
        if any(keyword in question_lower for keyword in schema_keywords):
            return False  # Schema questions can be answered conversationally
            
        # Check for conversation patterns
        if any(keyword in question_lower for keyword in conversation_keywords):
            # If it also has SQL keywords, it might be asking how to query something
            if any(keyword in question_lower for keyword in sql_keywords):
                # If it contains "how to" or similar, it's likely asking about SQL, not for SQL
                return not any(phrase in question_lower for phrase in ["how to", "how do i", "how can i", "explain how"])
            return False
            
        # Check for SQL patterns
        if any(keyword in question_lower for keyword in sql_keywords):
            return True
            
        # Default to conversational for ambiguous cases
        return False

    def plan_queries(self, question: str) -> Dict[str, Any]:
        """
        Plan the queries needed for a complex analysis question
        
        Args:
            question: The natural language question
            
        Returns:
            Dictionary with query planning details
        """
        # Prepare schema context if not already prepared
        self._prepare_schema_context()
        
        # Default query plan to return in case of errors
        default_plan = {
            "is_multi_query": False,
            "query_plan": [
                {
                    "query_name": "Default query",
                    "description": "Single query to answer the question",
                    "time_period": "current",
                    "key_metrics": []
                }
            ],
            "analysis_approach": "Direct single query approach"
        }
        
        # Prepare params for query planning
        params = {
            "schema": self.schema_context,
            "question": question
        }
        
        # Add memory if enabled
        if self.use_memory:
            memory_context = self._prepare_memory_for_query(question) or ""
            params["memory"] = memory_context
        
        try:
            # Generate query plan
            response = self.query_planner_chain.invoke(params)
            
            response_text = self._extract_response_content(response).strip()
                
            # Parse the simple key-value format
            needs_multiple = False
            num_queries = 1
            query_plan = []
            analysis_approach = "Direct query approach"
            
            # Extract information using regex patterns
            needs_multiple_match = re.search(r'NEEDS_MULTIPLE_QUERIES:\s*(yes|no|true|false)', response_text, re.IGNORECASE)
            if needs_multiple_match:
                value = needs_multiple_match.group(1).lower()
                needs_multiple = value == "yes" or value == "true"
            
            num_queries_match = re.search(r'NUMBER_OF_QUERIES_NEEDED:\s*(\d+)', response_text, re.IGNORECASE)
            if num_queries_match:
                try:
                    num_queries = int(num_queries_match.group(1))
                except ValueError:
                    num_queries = 1
            
            # Extract query descriptions
            query_matches = re.findall(r'-\s*Query\s+\d+:\s*(.+?)(?=\n-|\nANALYSIS_APPROACH:|$)', response_text, re.DOTALL)
            if query_matches:
                for i, description in enumerate(query_matches):
                    query_plan.append({
                        "query_name": f"Query {i+1}",
                        "description": description.strip(),
                        "time_period": "not specified",
                        "key_metrics": []
                    })
            else:
                # Fallback to default query plan
                query_plan = default_plan["query_plan"]
            
            # Extract analysis approach
            analysis_match = re.search(r'ANALYSIS_APPROACH:\s*(.+?)$', response_text, re.DOTALL)
            if analysis_match:
                analysis_approach = analysis_match.group(1).strip()
            
            # Build the final query plan
            result = {
                "is_multi_query": needs_multiple and num_queries > 1,
                "query_plan": query_plan,
                "analysis_approach": analysis_approach
            }
            
            print("Query plan generated successfully")
            return result
            
            print("No valid text in response, using default plan")
            return default_plan
                
        except Exception as e:
            print(f"Error planning queries: {e}")
            return default_plan
    
    def generate_sql_for_subquery(self, question: str, query_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SQL for a subquery in a multi-query plan
        
        Args:
            question: The original question
            query_info: Information about this specific query
            
        Returns:
            Dictionary with SQL generation details
        """
        # Create a modified question that focuses on this specific subquery
        query_name = query_info.get("query_name", "")
        description = query_info.get("description", "")
        time_period = query_info.get("time_period", "")
        
        # Construct a more specific question for this subquery
        specific_question = f"For {query_name}: {description}"
        if time_period:
            specific_question += f" for {time_period}"
        
        # Add the original question for context
        specific_question += f". This is part of answering: {question}"
        
        # Generate SQL for this specific question
        return self.generate_sql(specific_question)
    
    def execute_multi_query_analysis(self, question: str, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute multiple queries for complex analysis and provide a consolidated response
        
        Args:
            question: The natural language question
            query_plan: The query plan with multiple queries
            
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        query_results = []
        tables_info = []
        
        print(f"Starting multi-query analysis for: {question}")
        print(f"Query plan has {len(query_plan.get('query_plan', []))} queries planned")
        
        # Step 1: Generate and execute all SQL queries
        for i, query_info in enumerate(query_plan.get("query_plan", [])):
            print(f"Processing query {i+1}: {query_info.get('query_name', f'Query {i+1}')}")
            
            # Generate SQL for this specific query
            sql_generation = self.generate_sql_for_subquery(question, query_info)
            
            if not sql_generation["success"] or not sql_generation.get("sql"):
                print(f"Failed to generate SQL for query {i+1}")
                continue
                
            sql = sql_generation["sql"]
            print(f"Generated SQL for query {i+1}: {sql[:100]}...")  # Print first 100 chars for debug
            
            # Execute the SQL
            success, results, error = self.db_analyzer.execute_query(sql)
            
            if not success:
                print(f"Error executing query {i+1}: {error}")
                continue
                
            if not results or len(results) == 0:
                print(f"Query {i+1} returned no results")
                continue
            
            # Store the successful query and results
            query_name = query_info.get("query_name", f"Query {i+1}")
            query_results.append({
                "query_name": query_name,
                "sql": sql,
                "results": results,
                "row_count": len(results),
                "description": query_info.get("description", "")
            })
            
            print(f"Successfully executed query {i+1}, returned {len(results)} rows")
        
        # If no successful queries, return error
        if not query_results:
            print("No successful queries executed")
            return {
                "success": False,
                "question": question,
                "error": "Failed to execute any queries in the analysis plan",
                "execution_time": time.time() - start_time
            }
        
        # Step 2: Process and format each result table for the LLM, handling token limits
        print("Preparing query results for analysis")
        for qr in query_results:
            results = qr["results"]
            query_name = qr["query_name"]
            
            # Format the table with appropriate sampling if needed
            if len(results) > 100:  # For large result sets
                # Include statistics about the results
                import pandas as pd
                import numpy as np
                
                # Convert to DataFrame for easier analysis
                df = pd.DataFrame(results)
                
                # Get basic statistics
                stats = {}
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    stats[col] = {
                        "min": df[col].min() if not pd.isna(df[col].min()) else "N/A",
                        "max": df[col].max() if not pd.isna(df[col].max()) else "N/A",
                        "mean": df[col].mean() if not pd.isna(df[col].mean()) else "N/A",
                        "median": df[col].median() if not pd.isna(df[col].median()) else "N/A"
                    }
                
                # Sample rows for better representation
                sampled_rows = []
                
                # First few rows
                sampled_rows.extend(results[:20])
                
                # Last few rows
                if len(results) > 40:
                    sampled_rows.append({"note": "... skipping middle rows ..."})
                    sampled_rows.extend(results[-20:])
                
                # Format for the LLM
                table_formatted = (
                    f"LARGE RESULT SET: {len(results)} rows total, showing first 20 and last 20 rows.\n\n"
                    f"STATISTICS FOR NUMERIC COLUMNS:\n{json.dumps(stats, indent=2)}\n\n"
                    f"SAMPLED ROWS:\n{self._format_results_for_display(sampled_rows)}"
                )
            else:
                # For smaller result sets, include all rows
                table_formatted = self._format_results_for_display(results)
            
            # Add formatted table to tables_info
            tables_info.append(
                f"### {query_name} ###\n"
                f"DESCRIPTION: {qr.get('description', '')}\n"
                f"SQL:\n```sql\n{qr['sql']}\n```\n\n"
                f"RESULTS ({qr['row_count']} rows):\n{table_formatted}\n"
            )
        
        # Step 3: Generate analysis from the combined results
        print("Generating analysis from query results")
        tables_info_text = "\n\n".join(tables_info)
        
        # Prepare params for analysis
        params = {
            "schema": self.schema_context,
            "question": question,
            "tables_info": tables_info_text
        }
        
        # Add memory if enabled
        if self.use_memory:
            memory_context = self._prepare_memory_for_query(question) or ""
            params["memory"] = memory_context
        
        # Generate the analysis
        analysis_response = self.analysis_chain.invoke(params)
        
        analysis_text = self._extract_response_content(analysis_response).strip()
        print("Analysis generated successfully")
        
        # Step 4: Store in memory and session context
        if self.use_memory:
            # Store the analysis
            self._store_text_in_memory(question, analysis_text)
            
            # Also store each individual query
            for qr in query_results:
                self._store_in_memory(
                    f"Subquery for: {question} - {qr['query_name']}", 
                    qr['sql'], 
                    qr['results']
                )
        
        # Store in session context for future reference
        self.session_context["multi_query_results"] = query_results
        self.session_context["text_responses"].append({
            "question": question,
            "text": analysis_text,
            "timestamp": time.time()
        })
        
        print(f"Multi-query analysis completed in {time.time() - start_time:.2f} seconds")
        
        # Step 5: Return the full result with all tables
        return {
            "success": True,
            "question": question,
            "is_multi_query": True,
            "tables": query_results,
            "text": analysis_text,
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
        # Check if this is a "why" question requiring specialized causal analysis
        if self._is_why_question(question):
            print("Detected 'why' question, using specialized causal analysis approach")
            return self.handle_why_question(question)
        
        # Check if this is an analysis question requiring multiple queries
        elif self._is_analysis_question(question):
            # Plan the queries needed
            query_plan = self.plan_queries(question)
            
            if query_plan.get("is_multi_query", False):
                # Execute multi-query analysis
                return self.execute_multi_query_analysis(question, query_plan)
        
        # Check if this is a conversational question not requiring SQL
        # Use LLM-based classification instead of keyword matching
        if self._is_conversational_question(question):
            # Generate text response without SQL
            text_result = self.generate_text_response(question)
            return {
                "success": True,
                "question": question,
                "sql": None,
                "is_conversational": True,
                "source": "ai",
                "confidence": 90,
                "results": None,
                "text": text_result.get("text", "I couldn't generate a response.")
            }

        # Generate the SQL for a standard query
        generation_result = self.generate_sql(question)
        
        if not generation_result["success"] or not generation_result.get("sql"):
            # Generate text response for the error case
            text_result = self.generate_text_response(question)
            return {
                "success": False,
                "question": question,
                "sql": None,
                "source": generation_result.get("source", "unknown"),
                "confidence": generation_result.get("confidence", 0),
                "error": generation_result.get("error", "Failed to generate SQL"),
                "execution_time": generation_result.get("execution_time", 0),
                "results": None,
                "text": text_result.get("text", "I couldn't generate SQL for your question.")
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
        
        # Handle pagination for large result sets
        page_size = 10
        total_rows = len(results) if results else 0
        paginated_results = None
        table_id = None
        
        if success and results and len(results) > page_size:
            # Generate a unique table ID
            import uuid
            table_id = str(uuid.uuid4())
            
            # Store the full results for later pagination
            self.paginated_results[table_id] = {
                "question": question,
                "sql": sql,
                "results": results,
                "total_rows": total_rows,
                "timestamp": time.time()
            }
            
            # Return only the first page
            paginated_results = results[:page_size]
        else:
            paginated_results = results
        
        # Generate text response based on SQL and results
        text_result = self.generate_text_response(question, sql, results if success else None)
        
        # Store in memory if successful or even if failed (to remember errors too)
        if self.use_memory:
            self._store_in_memory(question, sql, results if success else None)
            
            # Update session context
            if success:
                self._update_session_context(question, sql, results)
            
        response = {
            "success": success,
            "question": question,
            "sql": sql,
            "is_conversational": False,
            "source": generation_result.get("source"),
            "confidence": generation_result.get("confidence"),
            "error": error,
            "auto_fixed": attempts > 0 and success,
            "fix_attempts": attempts,
            "execution_time": generation_result.get("execution_time", 0),
            "results": paginated_results,
            "text": text_result.get("text", "")
        }
        
        # Add pagination metadata if applicable
        if table_id:
            response["pagination"] = {
                "table_id": table_id,
                "total_rows": total_rows,
                "page_size": page_size,
                "current_page": 1,
                "total_pages": (total_rows + page_size - 1) // page_size
            }
        
        return response

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
                    # Ensure vector_memory is a string
                    if not isinstance(vector_memory, str):
                        vector_memory = str(vector_memory)
                    
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

    def get_paginated_results(self, table_id: str, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """
        Get a specific page of results for a previously executed query
        
        Args:
            table_id: The unique ID of the stored result set
            page: The page number to retrieve (1-indexed)
            page_size: Number of rows per page
            
        Returns:
            Dictionary with the requested page of results and pagination metadata
        """
        # Check if the table_id exists
        if table_id not in self.paginated_results:
            return {
                "success": False,
                "error": f"No results found for table_id: {table_id}",
                "results": []
            }
            
        # Get the stored results
        stored_data = self.paginated_results[table_id]
        results = stored_data["results"]
        total_rows = len(results)
        total_pages = (total_rows + page_size - 1) // page_size
        
        # Validate the page number
        if page < 1:
            page = 1
        elif page > total_pages:
            page = total_pages
            
        # Calculate start and end indices
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        
        # Get the requested page of results
        page_results = results[start_idx:end_idx]
        
        return {
            "success": True,
            "question": stored_data["question"],
            "sql": stored_data["sql"],
            "results": page_results,
            "pagination": {
                "table_id": table_id,
                "total_rows": total_rows,
                "page_size": page_size,
                "current_page": page,
                "total_pages": total_pages
            }
        }

    def _is_conversational_question(self, question: str) -> bool:
        """
        Use the LLM to determine if a question is conversational rather than requiring SQL
        
        Args:
            question: The natural language question
            
        Returns:
            True if the question is conversational, False if it likely requires SQL
        """
        # Prepare schema context if not already prepared
        self._prepare_schema_context()
        
        # Define a prompt template for determining if a question is conversational
        conversation_classifier_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""You are an expert at classifying database questions. Your task is to determine if a question requires SQL generation or if it's a conversational question that should be answered directly.

### DATABASE SCHEMA:
{schema}

### USER QUESTION:
"{question}"

### TASK:
Determine if this question requires generating SQL to query the database, or if it's a conversational question that should be answered directly.

Examples of SQL questions:
- "Show me the top 5 customers by sales"
- "List all orders from January 2023"
- "What was the total revenue last quarter?"
- "How many products are in each category?"

Examples of conversational questions:
- "What is a database index?"
- "How do I optimize SQL queries?"
- "Tell me about your capabilities"
- "What LLM are you using?"
- "How does this system work?"
- "What programming language is this built with?"
- "Can you explain what a foreign key is?"

### CLASSIFICATION:
Provide ONLY ONE of the following responses:
SQL_QUESTION - if the question requires database querying
CONVERSATIONAL - if the question is asking for information, explanation, or conversation

"""
        )
        
        # Create a temporary chain for classification
        conversation_classifier_chain = conversation_classifier_prompt | self.llm
        
        try:
            # Prepare params for classification
            params = {
                "schema": self.schema_context,
                "question": question
            }
            
            # Generate classification
            response = conversation_classifier_chain.invoke(params)
            
            classification = self._extract_response_content(response).strip().upper()
            
            # Check for conversational classification
            if "CONVERSATIONAL" in classification:
                print(f"LLM classified question as conversational: {question}")
                return True
            else:
                print(f"LLM classified question as requiring SQL: {question}")
                return False
            
            # Default to the keyword-based approach if LLM classification fails
            print("LLM classification failed, falling back to keyword-based approach")
            return not self._is_sql_question(question)
            
        except Exception as e:
            print(f"Error in conversational classification: {e}")
            # Fall back to the keyword-based approach
            return not self._is_sql_question(question)


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