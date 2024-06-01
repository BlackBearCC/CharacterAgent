from sqlalchemy.orm import scoped_session
from .models import Entity

class EntityMemory:
    def __init__(self, session: scoped_session):
        self.session = session

    def add_entity(self, entity: Entity):
        """Add a new entity to the database."""
        self.session.add(entity)
        self.session.commit()

    def get_entities(self, user_guid: str, count: int = 100):
        """Retrieve the latest entities for a given user GUID."""
        return (self.session.query(Entity)
                .filter_by(user_guid=user_guid)
                .order_by(Entity.entity_id.desc())
                .limit(count)
                .all())

    def get_entity_by_id(self, entity_id: int):
        """Retrieve a specific entity by its ID."""
        return self.session.query(Entity).filter_by(entity_id=entity_id).first()

    def update_entity(self, entity_id: int, **kwargs):
        """Update specific fields of an existing entity."""
        entity = self.session.query(Entity).filter_by(entity_id=entity_id).first()
        if entity:
            for key, value in kwargs.items():
                setattr(entity, key, value)
            self.session.commit()
            return entity
        return None

    def delete_entity(self, entity_id: int):
        """Delete a specific entity by its ID."""
        entity = self.session.query(Entity).filter_by(entity_id=entity_id).first()
        if entity:
            self.session.delete(entity)
            self.session.commit()

    # 实体处理逻辑...
