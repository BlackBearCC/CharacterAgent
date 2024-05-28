from sqlalchemy import Column, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid

from data.database.mysql.base import Base


class User(Base):
    """Represents a user entity."""
    __tablename__ = 'user'

    guid = Column(String(128), primary_key=True)
    username = Column(String(50))
    email = Column(String(100))
    game_uid = Column(String(16))

class UserDatabase:
    """Handles user-related database operations."""

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string, echo=False)
        self.Session = sessionmaker(bind=self.engine)
        self.Base = Base
        self.create_user_table()

    def create_user_table(self) -> None:
        """Create the user table if it does not exist."""
        self.Base.metadata.create_all(self.engine)

    def generate_guid(self) -> str:
        """Generate a unique GUID."""
        return str(uuid.uuid4())

    def add_user(self,username: str=None, email: str=None ) -> str:
        """Add a new user to the database."""
        new_guid = self.generate_guid()
        new_user = User(guid=new_guid, username=username, email=email)
        with self.Session() as session:
            session.add(new_user)
            session.commit()
        return new_guid

    def add_game_user(self, game_uid: str, username: str = None, email: str = None) -> str:
        """Add a new user to the database or update an existing one by game_id."""


        with self.Session() as session:
            existing_user = session.query(User).filter_by(game_uid=game_uid).first()

            if existing_user:
                # 用户已存在，
                return f"用户已存在,可使用/game/chat请求路径，uid:{game_uid}"
            else:
                # 如果用户不存在，创建新用户
                new_guid = self.generate_guid()
                new_user = User(guid=new_guid, username=username, email=email, game_uid=game_uid)
                session.add(new_user)
                session.commit()
                return f"新游戏用户已创建，游戏端/game/chat请求路径,uid:{game_uid}；通用/chat请求路径,uid:{new_guid}"


    def get_user_by_guid(self, guid: str) -> User:
        """Get a user by their GUID."""
        with self.Session() as session:
            return session.query(User).filter_by(guid=guid).first()

    def get_user_by_game_uid(self, game_uid: str) -> User:
        """Get a user by their GUID."""
        with self.Session() as session:
            return session.query(User).filter_by(game_uid=game_uid).first()

    def update_user(self, guid: str, username: str, email: str) -> None:
        """Update a user's information."""
        with self.Session() as session:
            user = session.query(User).filter_by(guid=guid).first()
            if user:
                user.username = username
                user.email = email
                session.commit()
