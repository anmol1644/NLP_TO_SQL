import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import google.generativeai as genai
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from database import Database


class QueryTemplate:
    def __init__(self, name: str, patterns: List[str], sql: str, requires_params: bool = False):
        """
        Initialize a query template
        
        Args:
            name: Template name
            patterns: List of pattern strings that match this template
            sql: SQL query template (may contain {placeholders})
            requires_params: Whether this template requires parameters to be extracted
        """
        self.name = name
        self.patterns = patterns
        self.sql = sql
        self.requires_params = requires_params


class ConsistentAIEngine:
    def __init__(self, db: Database, cache_file: str = "query_cache.json"):
        """
        Initialize the AI engine
        
        Args:
            db: Database instance
            cache_file: Path to the query cache file
        """
        load_dotenv()  # Load environment variables from .env file
        self.db = db
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
        # Initialize templates
        self._initialize_templates()
        
        # Initialize Google Gemini and LangChain
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Define SQL generation prompt
        self.sql_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""You are a SQL expert. I have a database with the following schema:

{schema}

Please write a PostgreSQL SQL query to answer this business question:
{question}

Follow these guidelines:
1. Use consistent column aliases (e.g., total_revenue, customer_name)
2. For financial queries, always include WHERE status = 'completed'
3. Use PostgreSQL date functions like DATE_TRUNC, EXTRACT instead of SQLite's strftime
4. Set LIMIT 10 unless another limit is specified
5. NEVER use dangerous SQL operations (DROP, DELETE, UPDATE, INSERT, ALTER, etc.)
6. Return ONLY the SQL query without any explanation or markdown formatting
7. Only use tables and columns that exist in the schema above
"""
        )
        
        # Initialize LangChain with Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0,
            google_api_key=self.api_key,
        )
        
        self.sql_chain = LLMChain(llm=self.llm, prompt=self.sql_prompt)
    
    def _initialize_templates(self):
        """Initialize query templates"""
        self.templates = [
            QueryTemplate(
                name="top_customers_by_revenue",
                patterns=[
                    "top customers by revenue",
                    "best customers by sales",
                    "highest spending customers",
                    "customers with most revenue",
                    "who are the top customers"
                ],
                sql="""SELECT 
    c.name as customer_name, 
    COUNT(o.id) as order_count, 
    SUM(o.total_amount) as total_revenue 
FROM customers c 
JOIN orders o ON c.id = o.customer_id 
WHERE o.status = 'completed' 
GROUP BY c.id 
ORDER BY total_revenue DESC 
LIMIT {limit}""",
                requires_params=True
            ),
            QueryTemplate(
                name="sales_by_month",
                patterns=[
                    "sales by month",
                    "monthly sales",
                    "revenue by month",
                    "monthly revenue",
                    "sales per month"
                ],
                sql="""SELECT 
    DATE_TRUNC('month', o.order_date) as month, 
    COUNT(o.id) as order_count, 
    SUM(o.total_amount) as total_revenue 
FROM orders o 
WHERE o.status = 'completed' 
GROUP BY month 
ORDER BY month""",
                requires_params=False
            ),
            QueryTemplate(
                name="category_performance",
                patterns=[
                    "category performance",
                    "performance by category",
                    "best selling categories",
                    "top product categories",
                    "sales by category"
                ],
                sql="""SELECT 
    p.category, 
    COUNT(DISTINCT o.id) as order_count, 
    SUM(oi.quantity) as units_sold, 
    SUM(oi.quantity * oi.unit_price) as total_revenue 
FROM products p 
JOIN order_items oi ON p.id = oi.product_id 
JOIN orders o ON oi.order_id = o.id 
WHERE o.status = 'completed' 
GROUP BY p.category 
ORDER BY total_revenue DESC""",
                requires_params=False
            ),
            QueryTemplate(
                name="recent_orders",
                patterns=[
                    "recent orders",
                    "latest orders",
                    "newest orders",
                    "orders in the last",
                    "orders from the past"
                ],
                sql="""SELECT 
    o.id as order_id, 
    c.name as customer_name, 
    o.order_date, 
    o.total_amount, 
    o.status 
FROM orders o 
JOIN customers c ON o.customer_id = c.id 
ORDER BY o.order_date DESC 
LIMIT {limit}""",
                requires_params=True
            ),
            QueryTemplate(
                name="average_order_value",
                patterns=[
                    "average order value",
                    "average order amount",
                    "typical order value",
                    "mean order value",
                    "average purchase amount"
                ],
                sql="""SELECT 
    AVG(total_amount) as average_order_value 
FROM orders 
WHERE status = 'completed'""",
                requires_params=False
            ),
            QueryTemplate(
                name="customer_order_history",
                patterns=[
                    "customer order history",
                    "order history for customer",
                    "orders by customer",
                    "customer purchase history",
                    "history of customer orders"
                ],
                sql="""SELECT 
    o.id as order_id, 
    o.order_date, 
    o.total_amount, 
    o.status, 
    COUNT(oi.id) as num_items 
FROM orders o 
JOIN order_items oi ON o.id = oi.order_id 
WHERE o.customer_id = {customer_id} 
GROUP BY o.id 
ORDER BY o.order_date DESC""",
                requires_params=True
            )
        ]
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load query cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_cache(self):
        """Save query cache to disk"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def _get_question_hash(self, question: str) -> str:
        """Generate a hash for a question to use as cache key"""
        # Normalize the question: lowercase, remove punctuation, extra spaces
        normalized = re.sub(r'[^\w\s]', '', question.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _match_template(self, question: str) -> Tuple[Optional[QueryTemplate], int, Dict[str, Any]]:
        """
        Try to match the question against templates
        
        Returns:
            Tuple of (matched_template, confidence_score, extracted_params)
        """
        best_template = None
        best_score = 0
        extracted_params = {}
        
        # Extract potential parameters
        limit_match = re.search(r'\b(?:top|first|best) (\d+)\b', question.lower())
        limit = int(limit_match.group(1)) if limit_match else 10
        extracted_params['limit'] = limit
        
        # Try to find customer_id if mentioned
        customer_id_match = re.search(r'\bcustomer (?:id)? ?[#]?(\d+)\b', question.lower())
        if customer_id_match:
            extracted_params['customer_id'] = int(customer_id_match.group(1))
        
        for template in self.templates:
            for pattern in template.patterns:
                score = fuzz.token_set_ratio(question.lower(), pattern.lower())
                if score > best_score:
                    best_score = score
                    best_template = template
        
        return best_template, best_score, extracted_params
    
    def _validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL for basic safety and syntax
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for dangerous operations
        dangerous_patterns = [
            r'\bDROP\b',
            r'\bDELETE\b',
            r'\bUPDATE\b',
            r'\bINSERT\b',
            r'\bALTER\b',
            r'\bCREATE\b',
            r'\bTRUNCATE\b',
            r'\bEXEC\b',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return False, f"Query contains dangerous operation: {pattern.strip('\\b')}"
        
        # Basic syntax check
        if not re.search(r'\bSELECT\b', sql, re.IGNORECASE):
            return False, "Query must contain SELECT statement"
        
        # Check for unresolved placeholders
        if re.search(r'\{.*?\}', sql):
            return False, "Query contains unresolved placeholders"
        
        # More detailed syntax checking could be implemented here
        # But for this example we'll just check for basic errors
        
        return True, None
    
    def translate(self, question: str) -> Dict[str, Any]:
        """
        Translate a natural language question to SQL
        
        Args:
            question: Natural language question
        
        Returns:
            Dictionary with query details:
            - sql: Generated SQL query
            - source: 'template' or 'ai'
            - confidence: Confidence score
            - execution_time: Time taken to generate SQL (not including execution)
        """
        # Check cache first
        question_hash = self._get_question_hash(question)
        if question_hash in self.cache:
            cached_result = self.cache[question_hash]
            cached_result['source'] = f"{cached_result['source']} (cached)"
            return cached_result
        
        # Try to match against templates first
        template, confidence, params = self._match_template(question)
        
        if template and confidence >= 70:
            # Template matched with high confidence
            try:
                sql = template.sql
                
                # Apply parameters if required
                if template.requires_params:
                    try:
                        sql = sql.format(**params)
                    except KeyError as e:
                        # Missing parameter, skip this template
                        return {
                            'sql': None,
                            'source': 'template',
                            'confidence': confidence,
                            'error': f"Missing parameter in template: {str(e)}"
                        }
                
                # Validate SQL
                is_valid, error = self._validate_sql(sql)
                if not is_valid:
                    return {
                        'sql': None,
                        'source': 'template',
                        'confidence': confidence,
                        'error': error
                    }
                
                result = {
                    'sql': sql,
                    'source': 'template',
                    'confidence': confidence
                }
                
                # Cache the result
                self.cache[question_hash] = result
                self._save_cache()
                
                return result
            
            except KeyError as e:
                # Missing template parameter
                pass
        
        # Fall back to AI if no template match or template formatting failed
        schema = self.db.get_schema_details()
        
        try:
            # Use LangChain with Gemini to generate SQL
            response = self.sql_chain.invoke({"schema": schema, "question": question})
            
            if response and isinstance(response, dict) and "text" in response:
                sql_response = response["text"]
            else:
                # Handle the case where response doesn't have the expected structure
                return {
                    'sql': None,
                    'source': 'ai',
                    'confidence': 0,
                    'error': "AI returned an unexpected response format"
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
                    'sql': None,
                    'source': 'ai',
                    'confidence': 90,  # AI confidence is assumed high but not 100%
                    'error': error
                }
            
            result = {
                'sql': sql,
                'source': 'ai',
                'confidence': 90  # AI confidence is assumed high but not 100%
            }
            
            # Cache the result
            self.cache[question_hash] = result
            self._save_cache()
            
            return result
            
        except Exception as e:
            return {
                'sql': None,
                'source': 'ai',
                'confidence': 0,
                'error': str(e)
            }
    
    def execute(self, question: str) -> Dict[str, Any]:
        """
        Translate a question to SQL and execute it
        
        Args:
            question: Natural language question
        
        Returns:
            Dictionary with execution results
        """
        # Translate question to SQL
        translation = self.translate(question)
        
        # If SQL generation failed, return the error
        if 'error' in translation or translation['sql'] is None:
            return {
                'success': False,
                'question': question,
                'sql': None,
                'source': translation.get('source', 'unknown'),
                'confidence': translation.get('confidence', 0),
                'error': translation.get('error', 'Failed to generate SQL'),
                'results': None
            }
        
        # Execute SQL query
        sql = translation['sql']
        success, results, error = self.db.execute_query(sql)
        
        return {
            'success': success,
            'question': question,
            'sql': sql,
            'source': translation['source'],
            'confidence': translation['confidence'],
            'error': error,
            'results': results
        } 