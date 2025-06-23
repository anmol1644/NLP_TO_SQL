import os
import time
import pandas as pd
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from sqlalchemy import inspect, text

from database import Database
from ai_engine import ConsistentAIEngine
from test_queries import run_tests
from db_analyzer import DatabaseAnalyzer
from smart_sql import SmartSQLGenerator


def setup_environment() -> bool:
    """
    Set up the environment
    
    Returns:
        True if setup succeeded, False otherwise
    """
    load_dotenv()  # Load environment variables from .env file
    
    # Check for required environment variables
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it in a .env file.")
        return False
    
    return True


def get_db_connection():
    """
    Get database connection parameters from environment variables or user input
    
    Returns:
        Tuple of (db_name, username, password, host, port) if successful
    """
    # Try to get database connection parameters from environment variables
    db_name = os.getenv("DB_NAME")
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    
    # Ask for any missing parameters
    if not db_name:
        db_name = input("Database name: ")
    
    if not username:
        username = input("Database username: ")
    
    if not password:
        password = input("Database password: ")
    
    return db_name, username, password, host, port


def analyze_database(db_name, username, password, host, port) -> DatabaseAnalyzer:
    """
    Analyze the database schema
    
    Returns:
        DatabaseAnalyzer instance
    """
    print(f"\nAnalyzing database '{db_name}' on {host}:{port}...")
    db_analyzer = DatabaseAnalyzer(db_name, username, password, host, port)
    
    try:
        # Test connection
        with db_analyzer.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        
        print("✓ Connected to database successfully.\n")
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return None
    
    # Analyze the schema
    try:
        print("Analyzing database schema (this may take a moment)...")
        schema_info = db_analyzer.analyze_schema()
        
        # Print database summary
        tables = list(schema_info["tables"].keys())
        relationships = schema_info["relationships"]
        print(f"\nDatabase Summary:")
        print(f"- {len(tables)} tables found")
        print(f"- {len(relationships)} relationships identified")
        
        # Print tables found
        print("\nTables:")
        for table_name, table_info in schema_info["tables"].items():
            row_count = table_info["row_count"]
            print(f"- {table_name} ({row_count} rows)")
        
        return db_analyzer
        
    except Exception as e:
        print(f"❌ Error analyzing schema: {str(e)}")
        return None


def interactive_mode(sql_generator: SmartSQLGenerator):
    """
    Run interactive mode for natural language queries
    
    Args:
        sql_generator: SmartSQLGenerator instance
    """
    print("\n=== Interactive Natural Language Query Mode ===")
    print("Enter your business questions in natural language.")
    print("The system will convert them to SQL and execute the queries.")
    print("Type 'exit', 'quit', or 'q' to return to the main menu.")
    print("Type 'help' to see example queries.")
    
    while True:
        # Get user input
        query = input("\nEnter your question: ")
        
        # Check for exit command
        if query.lower() in ['exit', 'quit', 'q']:
            print("Exiting interactive mode.")
            break
        
        # Check for help command
        if query.lower() in ['help', '?']:
            print("\nExample questions you can ask:")
            print("- Show me all table names in the database")
            print("- What are the top 10 customers by order value?")
            print("- How many products are in each category?")
            print("- What were our monthly sales in 2023?")
            print("- Show me customers who haven't placed orders in the last 3 months")
            print("- What is the average order value for each customer segment?")
            continue
        
        if not query.strip():
            continue
        
        # Process the query
        start_time = time.time()
        print("\nProcessing your question...")
        
        # Generate SQL and execute
        result = sql_generator.execute_query(query)
        elapsed_time = time.time() - start_time
        
        # Display results
        if result["success"]:
            print(f"\n✓ Query executed successfully in {elapsed_time:.2f} seconds")
            print(f"\nGenerated SQL:")
            print(f"{result['sql']}")
            
            if result.get("auto_fixed"):
                print(f"\n(Query was automatically fixed after {result['fix_attempts']} attempts)")
            
            if result["results"]:
                df = pd.DataFrame(result["results"])
                row_count = len(df)
                
                print(f"\nResults: {row_count} rows returned")
                if row_count > 0:
                    # Show all columns but limit rows
                    with pd.option_context('display.max_columns', None, 'display.width', 1000):
                        preview_rows = min(10, row_count)
                        print(df.head(preview_rows))
                    
                    if row_count > preview_rows:
                        print(f"... {row_count - preview_rows} more rows")
                    
                    # Ask if user wants to see all data or save to CSV
                    if row_count > 10:
                        while True:
                            option = input("\nOptions: [A]ll rows, [S]ave to CSV, [C]ontinue: ").lower()
                            
                            if option == 'a':
                                # Show all rows
                                with pd.option_context('display.max_columns', None, 'display.width', 1000):
                                    print(df)
                                break
                            elif option == 's':
                                # Save to CSV
                                csv_file = input("Enter filename to save (default: results.csv): ").strip() or "results.csv"
                                df.to_csv(csv_file, index=False)
                                print(f"Results saved to {csv_file}")
                                break
                            elif option == 'c':
                                break
            else:
                print("Query executed but returned no results.")
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
            print(f"\nPartial SQL query (if generated):")
            if result.get("sql"):
                print(result["sql"])


def demo_mode(sql_generator: SmartSQLGenerator):
    """
    Run a demonstration with predefined queries
    
    Args:
        sql_generator: SmartSQLGenerator instance
    """
    print("\n=== Demo Mode ===")
    print("Running a series of example queries to demonstrate the system...")
    
    # Define a set of common business questions to showcase
    demo_questions = [
        "What are all the tables in this database?",
        "Show me the first 5 rows from each table",
        "Which customers have placed the most orders?",
        "What's the total revenue per month over the last year?",
        "Which products have the highest sales volume?",
        "What's the average order value by customer segment?",
        "Show me the sales trend over time"
    ]
    
    for i, question in enumerate(demo_questions):
        print(f"\n\nDemo Query {i+1}: \"{question}\"")
        print("-" * 50)
        
        # Process the query
        result = sql_generator.execute_query(question)
        
        # Display results
        if result["success"]:
            print(f"\nGenerated SQL:")
            print(f"{result['sql']}")
            
            if result["results"]:
                df = pd.DataFrame(result["results"])
                print(f"\nResults: {len(df)} rows returned")
                if len(df) > 0:
                    with pd.option_context('display.max_columns', None, 'display.width', 1000, 'display.max_rows', 10):
                        print(df)
            else:
                print("\nQuery executed but returned no results.")
        else:
            print(f"\nError: {result.get('error', 'Unknown error')}")
        
        # If not the last question, pause before continuing
        if i < len(demo_questions) - 1:
            input("\nPress Enter to continue to the next demo query...")
    
    print("\nDemo completed.")


def main():
    """Main function"""
    print("Natural Language to SQL System")
    print("=============================\n")
    
    # Setup environment
    if not setup_environment():
        return
    
    # Get database connection parameters
    db_params = get_db_connection()
    if not db_params:
        return
    
    db_name, username, password, host, port = db_params
    
    # Analyze database
    db_analyzer = analyze_database(db_name, username, password, host, port)
    if not db_analyzer:
        return
    
    # Initialize SQL generator
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    sql_generator = SmartSQLGenerator(db_analyzer, model_name=model_name)
    
    # Main menu
    while True:
        print("\nMenu:")
        print("1. Interactive Query Mode")
        print("2. Run Demo Queries")
        print("3. Reanalyze Database")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == "1":
            interactive_mode(sql_generator)
        elif choice == "2":
            demo_mode(sql_generator)
        elif choice == "3":
            db_analyzer = analyze_database(db_name, username, password, host, port)
            if db_analyzer:
                # Reinitialize SQL generator with new schema
                sql_generator = SmartSQLGenerator(db_analyzer, model_name=model_name)
        elif choice == "0":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main() 