import os
import pandas as pd
from sqlalchemy import create_engine, inspect, MetaData, text, types
from sqlalchemy.ext.automap import automap_base
from typing import Dict, List, Any, Tuple, Optional


# Define a custom type for PostgreSQL XML
class PostgreSQLXML(types.UserDefinedType):
    def get_col_spec(self, **kw):
        return "XML"
    
    def bind_processor(self, dialect):
        def process(value):
            return value
        return process
    
    def result_processor(self, dialect, coltype):
        def process(value):
            return value
        return process


class DatabaseAnalyzer:
    """
    Analyzes a PostgreSQL database schema and provides detailed information
    about tables, columns, relationships, and data distributions
    """
    
    def __init__(
        self,
        db_name: str,
        username: str,
        password: str,
        host: str = "localhost",
        port: str = "5432"
    ):
        """
        Initialize the database analyzer
        
        Args:
            db_name: PostgreSQL database name
            username: PostgreSQL username
            password: PostgreSQL password
            host: PostgreSQL host
            port: PostgreSQL port
        """
        # Register PostgreSQL XML type
        from sqlalchemy.dialects import postgresql
        postgresql.base.ischema_names['xml'] = PostgreSQLXML
        
        self.db_name = db_name
        self.connection_string = f"postgresql://{username}:{password}@{host}:{port}/{db_name}"
        self.engine = create_engine(self.connection_string)
        self.metadata = MetaData()
        self.inspector = inspect(self.engine)
        self.schema_info = None
        
    def analyze_schema(self) -> Dict[str, Any]:
        """
        Analyze the database schema and return detailed information
        
        Returns:
            Dictionary containing schema information
        """
        print("Analyzing database schema...")
        schema = {}
        
        # Get all schemas
        with self.engine.connect() as connection:
            result = connection.execute(text("SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('information_schema', 'pg_catalog')"))
            schema_names = [row[0] for row in result]
        
        schema["schemas"] = schema_names
        schema["tables"] = {}
        
        # Table statistics
        with self.engine.connect() as connection:
            for schema_name in schema_names:
                table_names = self.inspector.get_table_names(schema=schema_name)
                for table_name in table_names:
                    table_info = self._get_table_info(table_name, connection, schema_name)
                    # Store with schema-qualified name
                    schema["tables"][f"{schema_name}.{table_name}"] = table_info
                    
                    # For backward compatibility, also store with just table name if it doesn't exist yet
                    # This ensures older code that access tables by just name continues to work
                    if table_name not in schema["tables"]:
                        schema["tables"][table_name] = table_info
        
        # Analyze relationships
        schema["relationships"] = self._analyze_relationships(schema_names)
        
        # Generate schema summary
        schema["summary"] = self._generate_schema_summary(schema)
        
        self.schema_info = schema
        return schema
    
    def _get_table_info(self, table_name: str, connection, schema_name: str = "public") -> Dict[str, Any]:
        """
        Get detailed information about a table
        
        Args:
            table_name: Name of the table
            connection: Database connection
            schema_name: Schema name (defaults to 'public')
            
        Returns:
            Dictionary with table details
        """
        table_info = {
            "schema": schema_name,
            "columns": [],
            "primary_key": self.inspector.get_pk_constraint(table_name, schema=schema_name).get('constrained_columns', []),
            "foreign_keys": [],
            "indexes": self.inspector.get_indexes(table_name, schema=schema_name),
            "row_count": 0,
            "sample_data": None
        }
        
        # Get column information
        for column in self.inspector.get_columns(table_name, schema=schema_name):
            column_info = {
                "name": column["name"],
                "type": str(column["type"]),
                "nullable": column.get("nullable", True),
                "default": str(column.get("default", "")),
                "primary_key": column["name"] in table_info["primary_key"],
                "stats": {}
            }
            
            table_info["columns"].append(column_info)
        
        # Get foreign keys
        for fk in self.inspector.get_foreign_keys(table_name, schema=schema_name):
            table_info["foreign_keys"].append({
                "constrained_columns": fk["constrained_columns"],
                "referred_table": fk["referred_table"],
                "referred_schema": fk.get("referred_schema", schema_name),
                "referred_columns": fk["referred_columns"]
            })
        
        # Get row count
        try:
            result = connection.execute(text(f"SELECT COUNT(*) FROM \"{schema_name}\".\"{table_name}\""))
            table_info["row_count"] = result.scalar()
        except Exception as e:
            print(f"Error getting row count for {schema_name}.{table_name}: {e}")
            table_info["row_count"] = "Error"
        
        # Get sample data (first 5 rows)
        try:
            result = connection.execute(text(f"SELECT * FROM \"{schema_name}\".\"{table_name}\" LIMIT 5"))
            rows = []
            for row in result:
                # Convert row to dictionary properly
                row_dict = {}
                for idx, column in enumerate(result.keys()):
                    row_dict[column] = row[idx]
                rows.append(row_dict)
            
            if rows:
                table_info["sample_data"] = rows
        except Exception as e:
            print(f"Error getting sample data for {schema_name}.{table_name}: {e}")
        
        # Get column statistics
        if table_info["row_count"] and table_info["row_count"] != "Error" and table_info["row_count"] > 0:
            self._analyze_column_statistics(table_name, table_info, connection, schema_name)
        
        return table_info
    
    def _analyze_column_statistics(self, table_name: str, table_info: Dict, connection, schema_name: str = "public") -> None:
        """
        Analyze statistics for columns in a table
        
        Args:
            table_name: Name of the table
            table_info: Table information dictionary to update
            connection: Database connection
            schema_name: Schema name (defaults to 'public')
        """
        for i, column in enumerate(table_info["columns"]):
            col_name = column["name"]
            col_type = column["type"].lower()
            
            # Initialize stats dictionary
            stats = {}
            
            try:
                # Special handling for XML columns
                if "xml" in col_type:
                    # Add XML-specific metadata
                    stats["is_xml"] = True
                    
                    # Count non-null values
                    result = connection.execute(text(
                        f"SELECT COUNT(*) FROM \"{schema_name}\".\"{table_name}\" "
                        f"WHERE \"{col_name}\" IS NOT NULL"
                    ))
                    non_null_count = result.scalar() or 0
                    stats["non_null_count"] = non_null_count
                    
                    # Calculate null percentage
                    result = connection.execute(text(
                        f"SELECT (COUNT(*) - COUNT(\"{col_name}\")) * 100.0 / COUNT(*) "
                        f"FROM \"{schema_name}\".\"{table_name}\""
                    ))
                    null_percentage = result.scalar() or 0
                    stats["null_percentage"] = null_percentage
                    
                    # Only attempt to analyze XML content if there's data
                    if non_null_count > 0:
                        # Get sample of the XML structure
                        result = connection.execute(text(
                            f"SELECT \"{col_name}\"::text FROM \"{schema_name}\".\"{table_name}\" "
                            f"WHERE \"{col_name}\" IS NOT NULL LIMIT 1"
                        ))
                        sample = result.scalar()
                        if sample:
                            # Store a truncated sample (first 200 chars)
                            stats["sample"] = sample[:200] + "..." if len(sample) > 200 else sample
                    
                # Skip other BLOB data types
                elif any(t in col_type for t in ["blob", "bytea", "json"]):
                    continue
                
                # Analyze based on column type for non-XML columns
                elif any(t in col_type for t in ["int", "double", "float", "numeric", "decimal"]):
                    # Numeric column analysis
                    result = connection.execute(text(
                        f"SELECT MIN(\"{col_name}\"), MAX(\"{col_name}\"), AVG(\"{col_name}\"), "
                        f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY \"{col_name}\") "
                        f"FROM \"{schema_name}\".\"{table_name}\" WHERE \"{col_name}\" IS NOT NULL"
                    ))
                    row = result.first()
                    if row:
                        stats["min"] = row[0]
                        stats["max"] = row[1]
                        stats["avg"] = row[2]
                        stats["median"] = row[3]
                
                    # Analyze null percentage for all columns
                    result = connection.execute(text(
                        f"SELECT (COUNT(*) - COUNT(\"{col_name}\")) * 100.0 / COUNT(*) "
                        f"FROM \"{schema_name}\".\"{table_name}\""
                    ))
                    null_percentage = result.scalar() or 0
                    stats["null_percentage"] = null_percentage
                
                # For categorical columns, get distinct value count and top 5 values
                elif any(t in col_type for t in ["char", "text", "enum", "bool"]) or "date" in col_type:
                    # Get distinct value count
                    result = connection.execute(text(
                        f"SELECT COUNT(DISTINCT \"{col_name}\") FROM \"{schema_name}\".\"{table_name}\""
                    ))
                    distinct_count = result.scalar() or 0
                    stats["distinct_count"] = distinct_count
                    
                    # Get top 5 most frequent values if not too many distinct values
                    if distinct_count > 0 and distinct_count < 1000:
                        result = connection.execute(text(
                            f"SELECT \"{col_name}\", COUNT(*) as count "
                            f"FROM \"{schema_name}\".\"{table_name}\" "
                            f"WHERE \"{col_name}\" IS NOT NULL "
                            f"GROUP BY \"{col_name}\" "
                            f"ORDER BY count DESC "
                            f"LIMIT 5"
                        ))
                        top_values = []
                        for row in result:
                            top_values.append({"value": str(row[0]), "count": row[1]})
                        stats["top_values"] = top_values
                    
                    # Analyze null percentage
                    result = connection.execute(text(
                        f"SELECT (COUNT(*) - COUNT(\"{col_name}\")) * 100.0 / COUNT(*) "
                        f"FROM \"{schema_name}\".\"{table_name}\""
                    ))
                    null_percentage = result.scalar() or 0
                    stats["null_percentage"] = null_percentage
                
                # Update column info with statistics
                table_info["columns"][i]["stats"] = stats
                
            except Exception as e:
                print(f"Error analyzing statistics for {schema_name}.{table_name}.{col_name}: {e}")
                # Set error in stats
                stats["error"] = str(e)
                table_info["columns"][i]["stats"] = stats
    
    def _analyze_relationships(self, schema_names: List[str] = ["public"]) -> List[Dict[str, Any]]:
        """
        Analyze relationships between tables in the database
        
        Args:
            schema_names: List of schema names to analyze
            
        Returns:
            List of relationships
        """
        relationships = []
        
        # Collect all foreign keys
        for schema_name in schema_names:
            for table_name in self.inspector.get_table_names(schema=schema_name):
                for fk in self.inspector.get_foreign_keys(table_name, schema=schema_name):
                    relationship = {
                        "source_schema": schema_name,
                        "source_table": table_name,
                        "source_columns": fk["constrained_columns"],
                        "target_schema": fk.get("referred_schema", schema_name),
                        "target_table": fk["referred_table"],
                        "target_columns": fk["referred_columns"],
                        "name": fk.get("name")
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def _generate_schema_summary(self, schema: Dict) -> str:
        """
        Generate a summary of the schema for the AI context
        
        Args:
            schema: The complete schema information
            
        Returns:
            A string summary of the schema
        """
        summary_parts = ["DATABASE SCHEMA SUMMARY:", ""]
        
        # Table summary
        summary_parts.append(f"Database: {self.db_name}")
        
        # Get list of unique tables (avoid duplicates from schema qualified and non-qualified names)
        seen_tables = set()
        unique_tables = []
        
        # First add tables without schema prefix
        for table_name, table_info in schema['tables'].items():
            if "." not in table_name:
                unique_tables.append((table_name, table_info))
                seen_tables.add(table_name.split(".")[-1])
        
        # Then add qualified tables that haven't been seen yet
        for table_name, table_info in schema['tables'].items():
            if "." in table_name:
                simple_name = table_name.split(".")[-1]
                if simple_name not in seen_tables:
                    unique_tables.append((table_name, table_info))
                    seen_tables.add(simple_name)
        
        summary_parts.append(f"Total tables: {len(unique_tables)}")
        summary_parts.append("")
        
        # Tables and columns
        summary_parts.append("TABLES:")
        for table_name, table_info in unique_tables:
            # Include schema name if available in the table_info
            display_name = table_name
            if "schema" in table_info and "." not in table_name:
                display_name = f"{table_info['schema']}.{table_name}"
                
            summary_parts.append(f"\nTable: {display_name} ({table_info['row_count']} rows)")
            
            # Primary key
            if table_info["primary_key"]:
                summary_parts.append(f"Primary Key: {', '.join(table_info['primary_key'])}")
            
            # Foreign keys
            if table_info["foreign_keys"]:
                for fk in table_info["foreign_keys"]:
                    source_cols = ', '.join(fk['constrained_columns'])
                    target_cols = ', '.join(fk['referred_columns'])
                    referred_table = fk['referred_table']
                    if 'referred_schema' in fk and fk['referred_schema'] != 'public':
                        referred_table = f"{fk['referred_schema']}.{referred_table}"
                    summary_parts.append(f"Foreign Key: {source_cols} -> {referred_table}({target_cols})")
            
            # Columns
            summary_parts.append("Columns:")
            for column in table_info["columns"]:
                nullable = "" if column["nullable"] else "NOT NULL"
                summary_parts.append(f"  - {column['name']} ({column['type']}) {nullable}")
                
                # Add stats if available
                if column["stats"]:
                    stats = column["stats"]
                    if "min" in stats and "max" in stats:
                        summary_parts.append(f"    Range: {stats['min']} to {stats['max']}")
                    if "distinct_count" in stats:
                        summary_parts.append(f"    Distinct values: {stats['distinct_count']}")
                    if "null_percentage" in stats and stats["null_percentage"] > 0:
                        summary_parts.append(f"    Null values: {stats['null_percentage']:.1f}%")
            
            # Sample data hint
            if table_info["sample_data"]:
                summary_parts.append("  (Sample data available)")
        
        # Relationships summary
        if schema['relationships']:
            summary_parts.append("\nRELATIONSHIPS:")
            for rel in schema['relationships']:
                source = f"{rel['source_schema']}.{rel['source_table']}({', '.join(rel['source_columns'])})"
                target = f"{rel['target_schema']}.{rel['target_table']}({', '.join(rel['target_columns'])})"
                summary_parts.append(f"  - {source} -> {target}")
        
        return "\n".join(summary_parts)
    
    def get_rich_schema_context(self) -> str:
        """
        Get a rich schema context for AI
        
        Returns:
            A detailed string representation of the schema context
        """
        if not self.schema_info:
            self.analyze_schema()
        
        context_parts = ["DATABASE SCHEMA:", ""]
        
        # Get the tables/views sorted by most important first
        sorted_tables = []
        for table_name, table_info in self.schema_info["tables"].items():
            if "." not in table_name:  # Use unqualified names only
                row_count = table_info["row_count"]
                
                # Skip system tables or empty tables
                if row_count == 0 or table_name.startswith(('pg_', 'sql_', 'information_schema')):
                    continue
                
                sorted_tables.append((table_name, table_info, row_count))
        
        # Sort by row count (most first)
        sorted_tables.sort(key=lambda x: x[2] if isinstance(x[2], int) else 0, reverse=True)
        
        # Generate table context
        for table_name, table_info, row_count in sorted_tables:
            # Table header
            if isinstance(row_count, int) and row_count > 0:
                context_parts.append(f"Table: {table_name} ({row_count:,} rows)")
            else:
                context_parts.append(f"Table: {table_name}")
            
            # Add columns
            context_parts.append("Columns:")
            for column in table_info["columns"]:
                col_name = column["name"]
                col_type = column["type"]
                pk_marker = "PK" if column["primary_key"] else ""
                nullable = "" if column["nullable"] else "NOT NULL"
                
                # Add special handling for XML column types
                if "xml" in col_type.lower():
                    context_parts.append(f"  - {col_name} (XML) {pk_marker} {nullable}")
                    # Add any XML-specific information if available in stats
                    if "stats" in column and column["stats"]:
                        stats = column["stats"]
                        if stats.get("sample"):
                            # Add a small sample of the XML structure
                            sample = stats["sample"]
                            # Truncate sample to a reasonable size
                            if len(sample) > 100:
                                sample = sample[:100] + "..."
                            context_parts.append(f"    [XML Sample Structure: {sample}]")
                else:
                    context_parts.append(f"  - {col_name} ({col_type}) {pk_marker} {nullable}")
                    
                # Add column stats for important columns (non-XML types)
                if "xml" not in col_type.lower() and "stats" in column and column["stats"]:
                    stats = column["stats"]
                    
                    # For numeric columns, show range
                    if "min" in stats and "max" in stats:
                        context_parts.append(f"    [Range: {stats['min']} to {stats['max']}]")
                    
                    # For categorical columns, show top values
                    if "top_values" in stats and stats["top_values"]:
                        top_values_str = ", ".join([f"{v['value']} ({v['count']})" for v in stats["top_values"][:3]])
                        context_parts.append(f"    [Top values: {top_values_str}]")
                    
                    # For all columns, show null percentage if significant
                    if "null_percentage" in stats and stats["null_percentage"] > 5:
                        context_parts.append(f"    [NULL: {stats['null_percentage']:.1f}%]")
            
            # Add foreign keys
            if table_info["foreign_keys"]:
                context_parts.append("Foreign Keys:")
                for fk in table_info["foreign_keys"]:
                    src_cols = ", ".join(fk["constrained_columns"])
                    ref_table = fk["referred_table"]
                    ref_cols = ", ".join(fk["referred_columns"])
                    context_parts.append(f"  - {src_cols} -> {ref_table}({ref_cols})")
            
            # Add indexes
            if table_info["indexes"]:
                context_parts.append("Indexes:")
                for idx in table_info["indexes"]:
                    idx_name = idx["name"]
                    idx_cols = ", ".join(idx["column_names"])
                    unique = "UNIQUE " if idx["unique"] else ""
                    context_parts.append(f"  - {idx_name}: {unique}({idx_cols})")
            
            # Add a sample row if available
            if table_info["sample_data"] and len(table_info["sample_data"]) > 0:
                context_parts.append("Sample Data:")
                sample = table_info["sample_data"][0]
                sample_items = []
                
                for col, val in sample.items():
                    # Don't include XML fields in the sample data output
                    if any(c["name"] == col and "xml" in c["type"].lower() for c in table_info["columns"]):
                        sample_items.append(f"{col}: [XML data]")
                    else:
                        # Truncate long values
                        if isinstance(val, str) and len(val) > 50:
                            val = val[:50] + "..."
                        sample_items.append(f"{col}: {val}")
                
                context_parts.append("  " + ", ".join(sample_items))
            
            # Add separator between tables
            context_parts.append("")
        
        # Add relationships
        if self.schema_info["relationships"]:
            context_parts.append("RELATIONSHIPS:")
            for rel in self.schema_info["relationships"]:
                src_table = rel["source_table"]
                src_cols = ", ".join(rel["source_columns"])
                tgt_table = rel["target_table"]
                tgt_cols = ", ".join(rel["target_columns"])
                
                context_parts.append(f"- {src_table}({src_cols}) -> {tgt_table}({tgt_cols})")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def execute_query(self, query: str) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Execute a SQL query and return the results
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple of (success, results, error_message)
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                
                if result.returns_rows:
                    # Convert result to a list of dictionaries
                    columns = result.keys()
                    data = []
                    for row in result:
                        row_dict = {}
                        for idx, column in enumerate(columns):
                            value = row[idx]
                            # Convert non-serializable types to strings for JSON compatibility
                            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                                value = str(value)
                            row_dict[column] = value
                        data.append(row_dict)
                    
                    return True, data, None
                else:
                    return True, [], None
                
        except Exception as e:
            return False, None, str(e)


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    
    load_dotenv()
    db_name = os.getenv("DB_NAME", "postgres")
    username = os.getenv("DB_USERNAME", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    
    analyzer = DatabaseAnalyzer(db_name, username, password, host, port)
    schema = analyzer.analyze_schema()
    
    # Print schema summary
    print(analyzer.get_rich_schema_context())
    
    # Example query
    with analyzer.engine.connect() as connection:
        result = connection.execute(text("SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('information_schema', 'pg_catalog')"))
        schema_names = [row[0] for row in result]
        print("\nAvailable schemas:")
        for schema_name in schema_names:
            print(f"- {schema_name}")
        
        print("\nTables in database:")
        for schema_name in schema_names:
            result = connection.execute(text(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema_name}'"))
            for row in result:
                print(f"- {schema_name}.{row['table_name']}") 