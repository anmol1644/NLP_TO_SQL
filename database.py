import os
import random
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Date, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import NUMERIC
from urllib.parse import quote_plus

# Base class for SQLAlchemy ORM
Base = declarative_base()


# Define the tables
class Customer(Base):
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    phone = Column(String)
    city = Column(String)
    registration_date = Column(Date, nullable=False)
    
    orders = relationship("Order", back_populates="customer")
    
    def __repr__(self):
        return f"<Customer(id={self.id}, name='{self.name}')>"


class Product(Base):
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    price = Column(NUMERIC(10, 2), nullable=False)
    stock_quantity = Column(Integer, nullable=False)
    
    order_items = relationship("OrderItem", back_populates="product")
    
    def __repr__(self):
        return f"<Product(id={self.id}, name='{self.name}', price=${self.price})>"


class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False)
    order_date = Column(Date, nullable=False)
    total_amount = Column(NUMERIC(10, 2), nullable=False)
    status = Column(String, nullable=False)
    
    customer = relationship("Customer", back_populates="orders")
    order_items = relationship("OrderItem", back_populates="order")
    
    def __repr__(self):
        return f"<Order(id={self.id}, customer_id={self.customer_id}, total_amount=${self.total_amount})>"


class OrderItem(Base):
    __tablename__ = 'order_items'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=False)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(NUMERIC(10, 2), nullable=False)
    
    order = relationship("Order", back_populates="order_items")
    product = relationship("Product", back_populates="order_items")
    
    def __repr__(self):
        return f"<OrderItem(id={self.id}, order_id={self.order_id}, product_id={self.product_id}, quantity={self.quantity})>"


class Database:
    def __init__(self, db_name: str = "ecommerce", 
                 username: str = "postgres", 
                 password: str = "postgres",
                 host: str = "localhost",
                 port: str = "5432"):
        """
        Initialize the database connection
        
        Args:
            db_name: PostgreSQL database name
            username: PostgreSQL username
            password: PostgreSQL password
            host: PostgreSQL host
            port: PostgreSQL port
        """
        self.db_name = db_name
        # URL encode the password to handle special characters
        encoded_password = quote_plus(password)
        self.connection_string = f"postgresql://{username}:{encoded_password}@{host}:{port}/{db_name}"
        try:
            self.engine = create_engine(self.connection_string)
            self.Session = sessionmaker(bind=self.engine)
        except Exception as e:
            print(f"Error creating database engine: {str(e)}")
            raise
        
    def create_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(self.engine)
    
    def drop_tables(self):
        """Drop all tables from the database"""
        Base.metadata.drop_all(self.engine)
    
    def reset_database(self):
        """Reset the database by dropping and recreating all tables"""
        self.drop_tables()
        self.create_tables()
    
    def generate_sample_data(self):
        """Generate and insert sample data into the database"""
        session = self.Session()
        
        # Generate customers
        print("Generating customers...")
        customers = []
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        
        for i in range(1, 1001):
            registration_date = datetime.now() - timedelta(days=random.randint(1, 1000))
            customer = Customer(
                name=f"Customer {i}",
                email=f"customer{i}@example.com",
                phone=f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                city=random.choice(cities),
                registration_date=registration_date
            )
            customers.append(customer)
        
        session.add_all(customers)
        session.commit()
        
        # Generate products
        print("Generating products...")
        products = []
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Toys', 
                     'Sports', 'Beauty', 'Automotive', 'Grocery', 'Office']
        
        for i in range(1, 501):
            product = Product(
                name=f"Product {i}",
                category=random.choice(categories),
                price=Decimal(str(round(random.uniform(10.0, 1000.0), 2))),
                stock_quantity=random.randint(0, 1000)
            )
            products.append(product)
        
        session.add_all(products)
        session.commit()
        
        # Generate orders and order items
        print("Generating orders and order items...")
        statuses = ['completed', 'processing', 'cancelled', 'shipped', 'pending']
        
        for i in range(1, 5001):
            # Random date within the last 3 years
            order_date = datetime.now() - timedelta(days=random.randint(1, 1095))
            
            # Create order
            order = Order(
                customer_id=random.randint(1, 1000),
                order_date=order_date,
                total_amount=Decimal('0.00'),  # Initialize as Decimal
                status=random.choice(statuses)
            )
            session.add(order)
            session.flush()  # To get the order.id
            
            # Create 1-5 order items for this order
            num_items = random.randint(1, 5)
            order_total = Decimal('0.00')  # Initialize as Decimal
            
            for _ in range(num_items):
                product_id = random.randint(1, 500)
                product = session.query(Product).get(product_id)
                quantity = random.randint(1, 5)
                unit_price = product.price  # This will be Decimal
                
                order_item = OrderItem(
                    order_id=order.id,
                    product_id=product_id,
                    quantity=quantity,
                    unit_price=unit_price
                )
                
                session.add(order_item)
                order_total += unit_price * quantity  # Both are Decimal or can be multiplied with Decimal
            
            # Update order total
            order.total_amount = order_total
            
            # Commit in batches to avoid memory issues
            if i % 100 == 0:
                session.commit()
                print(f"Generated {i} orders...")
        
        session.commit()
        print("Sample data generation complete.")
    
    def get_schema_details(self) -> str:
        """Get a description of the database schema for AI context"""
        inspector = inspect(self.engine)
        schema_info = []
        
        for table_name in inspector.get_table_names():
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append(f"- {column['name']} ({str(column['type'])})")
            
            foreign_keys = []
            for fk in inspector.get_foreign_keys(table_name):
                foreign_keys.append(f"- {', '.join(fk['constrained_columns'])} -> {fk['referred_table']}.{', '.join(fk['referred_columns'])}")
            
            table_info = [
                f"Table: {table_name}",
                "Columns:",
                *columns
            ]
            
            if foreign_keys:
                table_info.extend([
                    "Foreign Keys:",
                    *foreign_keys
                ])
            
            schema_info.append("\n".join(table_info))
        
        return "\n\n".join(schema_info)
    
    def execute_query(self, query: str) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Execute SQL query and return results as JSON-compatible data
        
        Returns:
        - Tuple of (success, results, error_message)
        - If success is True, results contains the query results
        - If success is False, error_message contains the error description
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                
                if result.returns_rows:
                    # Convert result to a list of dictionaries
                    columns = result.keys()
                    data = []
                    for row in result:
                        row_dict = {columns[i]: value for i, value in enumerate(row)}
                        # Convert non-serializable types to strings
                        for key, value in row_dict.items():
                            if isinstance(value, datetime):
                                row_dict[key] = value.isoformat()
                            elif isinstance(value, Decimal):
                                row_dict[key] = float(value)  # Convert Decimal to float for JSON compatibility
                        data.append(row_dict)
                    
                    return True, data, None
                else:
                    return True, [], None
                
        except Exception as e:
            return False, None, str(e)
    
    def execute_query_to_df(self, query: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Execute SQL query and return results as pandas DataFrame
        
        Returns:
        - Tuple of (success, dataframe, error_message)
        """
        success, results, error = self.execute_query(query)
        
        if success and results is not None:
            return True, pd.DataFrame(results), None
        else:
            return False, None, error


if __name__ == "__main__":
    # Test the database class
    db = Database()
    db.reset_database()
    db.generate_sample_data()
    
    # Print schema
    print(db.get_schema_details())
    
    # Test query
    query = "SELECT c.name as customer_name, COUNT(o.id) as order_count, SUM(o.total_amount) as total_revenue " \
            "FROM customers c JOIN orders o ON c.id = o.customer_id " \
            "WHERE o.status = 'completed' " \
            "GROUP BY c.id " \
            "ORDER BY total_revenue DESC " \
            "LIMIT 5"
    
    success, results, error = db.execute_query(query)
    
    if success:
        print("\nTop 5 customers by revenue:")
        for row in results:
            print(f"{row['customer_name']}: {row['order_count']} orders, ${row['total_revenue']:.2f} revenue")
    else:
        print(f"Error: {error}") 