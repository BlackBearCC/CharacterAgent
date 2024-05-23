from typing import Any, List
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, MetaData

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from data.database.mysql.base import Base


class Entity(Base):
    """Data model for entity."""
    __tablename__ = 'entity_store'
    entity_id = Column(Integer, primary_key=True)
    entity = Column(String(length=255))
    summary = Column(String(length=255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_guid = Column(String(128), ForeignKey('user.guid'))

def create_entity_model(table_name: str, Base: Any) -> Any:
    """Create EntityModel class dynamically."""
    class EntityModel(Base):
        __tablename__ = table_name
        __mapper_args__ = {'extend_existing': True}
        entity_id = Column(Integer, primary_key=True)
        entity = Column(String(length=255))
        summary = Column(String(length=255))
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        user_guid = Column(String(128), ForeignKey('user.guid'))
    return EntityModel

class EntityMemory:
    """Entity history stored in an SQL database."""

    def __init__(self, connection_string: str, table_name: str = "entity_store",
                 _create_table_if_not_exists: bool = True):
        self.engine = create_engine(connection_string, echo=False)
        self.table_name = table_name
        self.Session = sessionmaker(bind=self.engine)
        self.EntityModel = Entity
        if _create_table_if_not_exists:
            self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        """Create table if not exists."""
        metadata = MetaData(self.engine)
        if self.table_name not in metadata.tables:
            self.EntityModel.metadata.create_all(self.engine)

    def _create_entity_from_row(self, row: Any, entity_model_instance: Any) -> Entity:
        """Create Entity object from database row."""
        entity = entity_model_instance()
        for column_name, value in row.__dict__.items():
            if column_name != '_sa_instance_state':
                setattr(entity, column_name, value)
        return entity

    def _get_entities_query(self, user_guid: str) -> Any:
        """Get query for fetching entities."""
        return self.Session().query(self.EntityModel).filter_by(user_guid=user_guid)

    def get_entities(self, user_guid: str, count: int = 10) -> List[Entity]:
        """Get entities."""
        query = self._get_entities_query(user_guid).order_by(self.EntityModel.entity_id.desc()).limit(count)
        rows = query.all()
        columns = {c.key for c in self.EntityModel.__table__.columns}
        return [self._create_entity_from_row(row, self.EntityModel) for row in rows]

    def get_entity(self, user_guid: str) -> Entity:
        """Get entity."""
        query = self._get_entities_query(user_guid).order_by(self.EntityModel.entity_id.desc())
        row = query.first()
        if row is not None:
            columns = {c.key for c in self.EntityModel.__table__.columns}
            return self._create_entity_from_row(row, self.EntityModel)
        else:
            return None

    def save_entity(self, user_guid: str, data: Entity) -> None:
        """Save entity."""
        with self.Session() as session:
            if data.entity_id is not None:
                entity = session.query(self.EntityModel).filter_by(entity_id=data.entity_id).first()
                if entity:
                    entity.entity = data.entity
                    entity.summary = data.summary
                    session.commit()
                else:
                    raise ValueError("Entity ID not found.")
            else:
                new_entity = self.EntityModel(
                    user_guid=user_guid,
                    entity=data.entity,
                    summary=data.summary,
                    created_at=data.created_at,
                    updated_at=data.updated_at
                )
                session.add(new_entity)
                session.commit()
