# Natural Language to SQL Query System

A Python application that dynamically analyzes any PostgreSQL database schema and converts natural language business questions into SQL queries using AI.

## Features

- **Dynamic Database Analysis**: Automatically discovers and analyzes any PostgreSQL database schema
- **AI-Powered SQL Generation**: Uses Google Gemini to convert natural language to precise SQL
- **No Predefined Templates**: Works with any database structure without requiring pre-configured templates
- **Schema Exploration**: Analyzes tables, columns, relationships, and data distributions
- **Auto-Fix Capability**: Automatically attempts to fix SQL errors when they occur
- **Interactive Mode**: Ask natural language questions and view query results immediately
- **Demo Mode**: Runs example queries to showcase capabilities
- **Query Statistics**: Provides confidence scores and execution metrics

## Requirements

- Python 3.8+
- PostgreSQL database
- Google Gemini API key
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/nlp-to-sql.git
cd nlp-to-sql
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

Create a `.env` file in the project root with the following variables:

```
# Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# PostgreSQL Database Connection (optional, can be entered interactively)
DB_NAME=your_database_name
DB_USERNAME=your_database_username
DB_PASSWORD=your_database_password
DB_HOST=localhost
DB_PORT=5432

# Optional Gemini model selection (defaults to gemini-2.0-flash)
GEMINI_MODEL=gemini-2.0-flash
```

## Usage

Run the main application:

```bash
python main.py
```

The application will:
1. Connect to your PostgreSQL database
2. Automatically analyze and understand the database schema
3. Provide options for interactive natural language queries or demo mode

### Interactive Mode

In interactive mode, you can:
- Type natural language questions about your data
- View the generated SQL queries
- See the query results with formatting options
- Export query results to CSV files

Example questions:
- "Show me the top 10 customers by total order value"
- "What's the average sales per month over the last year?"
- "Which product categories have the highest profit margins?"
- "List all customers who haven't made a purchase in the last 6 months"

### Demo Mode

The demo mode runs a set of predefined queries to showcase the system's capabilities with your specific database. This is useful for exploring what kinds of questions the system can answer.

## How It Works

The system works in several steps:

1. **Database Analysis**: Connects to your PostgreSQL database and analyzes:
   - Table and column structures
   - Primary and foreign key relationships
   - Data distributions and statistics
   - Sample data patterns

2. **Schema Understanding**: Builds a comprehensive understanding of the database schema that the AI can use as context.

3. **Query Generation**: When you ask a question:
   - The system provides the AI with database context and your question
   - The AI generates a PostgreSQL-compatible SQL query
   - The system validates the query for safety and syntax

4. **Query Execution**: Executes the SQL against your database and returns the results.

5. **Error Handling**: If errors occur, the system can automatically attempt to fix the SQL and try again.

## Project Structure

- `main.py` - Main application entry point
- `db_analyzer.py` - Database schema analyzer
- `smart_sql.py` - AI-powered SQL generation engine

## Limitations

- The system is designed for SELECT queries only (no data modification)
- Complex analytical questions may require refinement
- Performance depends on database size and query complexity

## License

[MIT License](LICENSE) 