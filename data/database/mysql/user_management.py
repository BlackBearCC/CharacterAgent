import logging

from sqlalchemy.orm import scoped_session
from sqlalchemy.exc import SQLAlchemyError

import uuid

from data.database.mysql.models import User


class UserDatabase:
    """Handles user-related database operations."""

    def __init__(self, session: scoped_session):
        self.session = session

    def generate_guid(self) -> str:
        """Generate a unique GUID."""
        return str(uuid.uuid4())

    def add_user(self, username: str = None, email: str = None) -> str:
        """Add a new user to the database."""
        try:
            new_guid = self.generate_guid()
            new_user = User(guid=new_guid, username=username, email=email)
            self.session.add(new_user)
            self.session.commit()
            return new_guid
        except SQLAlchemyError as e:
            self.session.rollback()
            raise Exception(f"Error adding user: {e}")

    def add_game_user(self, game_uid: str, username: str = None, role_name: str = None, email: str = None) -> str:
        """Add a new user to the database or update an existing one by game_uid."""
        try:
            existing_user = self.session.query(User).filter_by(game_uid=game_uid).first()
            if existing_user:
                return f"用户已存在, 可使用 /game/chat 请求路径，uid: {game_uid}"
            else:
                new_guid = self.generate_guid()
                new_user = User(guid=new_guid, username=username, role_name=role_name, email=email, game_uid=game_uid)
                self.session.add(new_user)
                self.session.commit()
                return f"新游戏用户已创建，游戏端 /game/chat 请求路径, uid: {game_uid}; 通用 /chat 请求路径, uid: {new_guid}"
        except SQLAlchemyError as e:
            self.session.rollback()
            raise Exception(f"Error processing game user: {e}")

    def update_game_user(self, game_uid: str, new_user_name: str, new_role_name: str) -> str:
        """Update the username and role name of an existing game user by game UID."""
        try:
            with self.session() as session:
                user = session.query(User).filter_by(game_uid=game_uid).first()
                if user:
                    user.username = new_user_name
                    user.role_name = new_role_name
                    session.commit()
                    return f"用户已更新，新的用户名: {new_user_name}, 新的角色名: {new_role_name}"
                else:
                    return None  # 用户未找到
        except SQLAlchemyError as e:
            self.session.rollback()
            logging.error(f"尝试更新游戏用户时发生数据库错误: {e}")
            raise  # 继续向上抛出异常以便调用者可以处理

    def get_user_by_guid(self, guid: str) -> User:
        """Get a user by their GUID."""
        try:
            return self.session.query(User).filter_by(guid=guid).first()
        except SQLAlchemyError as e:
            raise Exception(f"Error retrieving user by GUID: {e}")

    def get_user_by_game_uid(self, game_uid: str) -> User:
        """Get a user by their game UID."""
        try:
            return self.session.query(User).filter_by(game_uid=game_uid).first()
        except SQLAlchemyError as e:
            raise Exception(f"Error retrieving user by game UID: {e}")
