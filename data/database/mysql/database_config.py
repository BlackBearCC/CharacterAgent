from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from .models import Base
import os



def setup_database(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(engine)
