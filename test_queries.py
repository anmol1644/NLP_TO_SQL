import os
import time
from typing import Dict, List, Any

import pandas as pd
from sqlalchemy import inspect

from ai_engine import ConsistentAIEngine
from database import Database


class QueryTester:
    def __init__(self, ai_engine: ConsistentAIEngine):
        """
        Initialize the query tester
        
        Args:
            ai_engine: ConsistentAIEngine instance
        """
        self.ai_engine = ai_engine
    
    def test_consistency(self, question_groups: List[List[str]], verbose: bool = True) -> bool:
        """
        Test consistency by verifying that variations of the same question produce identical SQL
        
        Args:
            question_groups: List of question groups, where each group contains variations of the same question
            verbose: Whether to print detailed results
        
        Returns:
            True if all tests pass, False otherwise
        """
        if verbose:
            print("\n===== CONSISTENCY TESTS =====\n")
        
        all_passed = True
        
        for i, group in enumerate(question_groups):
            if verbose:
                print(f"Test Group {i+1}: {group[0]}")
                print("-" * 40)
            
            # Get SQL for all questions in the group
            results = []
            for question in group:
                start_time = time.time()
                translation = self.ai_engine.translate(question)
                elapsed_time = time.time() - start_time
                
                results.append({
                    'question': question,
                    'sql': translation.get('sql'),
                    'source': translation.get('source'),
                    'confidence': translation.get('confidence'),
                    'error': translation.get('error'),
                    'time': elapsed_time
                })
                
                if verbose:
                    print(f"Q: {question}")
                    print(f"Source: {translation.get('source')}, Confidence: {translation.get('confidence')}")
                    print(f"SQL: {translation.get('sql')}")
                    print(f"Time: {elapsed_time:.3f}s")
                    print()
            
            # Check if all SQLs are identical (ignoring case and whitespace)
            normalized_sqls = []
            for result in results:
                if result['sql'] is None:
                    normalized_sqls.append(None)
                else:
                    # Normalize SQL: lowercase, remove extra spaces
                    normalized = " ".join(result['sql'].lower().split())
                    normalized_sqls.append(normalized)
            
            if len(set(normalized_sqls)) == 1 and normalized_sqls[0] is not None:
                if verbose:
                    print("✓ PASSED: All variations produced identical SQL\n")
            else:
                if verbose:
                    print("✗ FAILED: Variations produced different SQL or errors\n")
                all_passed = False
        
        return all_passed
    
    def test_business_questions(self, questions: List[str], execute: bool = True, verbose: bool = True) -> Dict[str, Any]:
        """
        Test real business questions
        
        Args:
            questions: List of business questions to test
            execute: Whether to execute the generated SQL queries
            verbose: Whether to print detailed results
        
        Returns:
            Dictionary with test results
        """
        if verbose:
            print("\n===== BUSINESS QUESTION TESTS =====\n")
        
        results = []
        
        for question in questions:
            if verbose:
                print(f"Question: {question}")
                print("-" * 40)
            
            # Translate and optionally execute the question
            start_time = time.time()
            
            if execute:
                result = self.ai_engine.execute(question)
                elapsed_time = time.time() - start_time
                
                # Add elapsed time to result
                result['time'] = elapsed_time
                
                if verbose:
                    self._print_execution_result(result)
            else:
                translation = self.ai_engine.translate(question)
                elapsed_time = time.time() - start_time
                
                result = {
                    'question': question,
                    'sql': translation.get('sql'),
                    'source': translation.get('source'),
                    'confidence': translation.get('confidence'),
                    'error': translation.get('error'),
                    'time': elapsed_time,
                    'success': 'sql' in translation and translation['sql'] is not None,
                    'results': None
                }
                
                if verbose:
                    self._print_translation_result(result)
            
            results.append(result)
        
        return {
            'total': len(results),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'results': results
        }
    
    def _print_translation_result(self, result: Dict[str, Any]):
        """Print a translation result nicely formatted"""
        if result['success']:
            print(f"✓ SUCCESS ({result['time']:.3f}s)")
            print(f"Source: {result['source']}, Confidence: {result['confidence']}")
            print(f"SQL: {result['sql']}\n")
        else:
            print(f"✗ FAILED ({result['time']:.3f}s)")
            print(f"Error: {result['error']}\n")
    
    def _print_execution_result(self, result: Dict[str, Any]):
        """Print an execution result nicely formatted"""
        if result['success']:
            print(f"✓ SUCCESS ({result['time']:.3f}s)")
            print(f"Source: {result['source']}, Confidence: {result['confidence']}")
            print(f"SQL: {result['sql']}")
            
            if result['results']:
                # Convert to DataFrame for nicer display
                df = pd.DataFrame(result['results'])
                # Display row count and sample data
                print(f"\nResult: {len(result['results'])} rows")
                print("\nSample data:")
                # Show at most 5 rows and limit column width for display
                with pd.option_context('display.max_columns', None, 'display.width', 1000, 'display.max_rows', 5):
                    print(df)
            else:
                print("\nResult: 0 rows")
            
            print()
        else:
            print(f"✗ FAILED ({result['time']:.3f}s)")
            print(f"Error: {result['error']}\n")


def run_tests(db_name: str = "ecommerce", verbose: bool = True):
    """
    Run all tests
    
    Args:
        db_name: PostgreSQL database name
        verbose: Whether to print detailed results
    
    Returns:
        Dictionary with test results
    """
    # Get database connection parameters from environment variables if available
    db_name = os.getenv("DB_NAME", db_name)
    username = os.getenv("DB_USERNAME", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")

    # Initialize database and AI engine
    db = Database(db_name=db_name, username=username, password=password, host=host, port=port)
    ai_engine = ConsistentAIEngine(db)
    
    # Initialize tester
    tester = QueryTester(ai_engine)
    
    # Define consistency test groups
    consistency_groups = [
        [
            "top customers by revenue", 
            "best customers by sales", 
            "highest spending customers",
            "who are the top customers"
        ],
        [
            "sales by month", 
            "monthly sales", 
            "revenue by month"
        ],
        [
            "category performance", 
            "performance by category", 
            "sales by category"
        ],
        [
            "average order value", 
            "average order amount", 
            "typical order value"
        ]
    ]
    
    # Define business questions
    business_questions = [
        "show me the top 10 customers by revenue",
        "what were our monthly sales for the last year",
        "which product categories perform best",
        "what is our average order value",
        "show recent orders with status",
        "how many orders were placed in each city",
        "which products have less than 50 items in stock",
        "what percentage of orders are shipped vs pending",
        "who are our top 5 customers in Chicago",
        "what's the total revenue for each month in 2023"
    ]
    
    # Run consistency tests
    consistency_passed = tester.test_consistency(consistency_groups, verbose=verbose)
    
    # Run business question tests
    business_results = tester.test_business_questions(business_questions, execute=True, verbose=verbose)
    
    # Return summary results
    return {
        'consistency_passed': consistency_passed,
        'business_questions': business_results
    }


if __name__ == "__main__":
    # Run tests
    run_tests() 