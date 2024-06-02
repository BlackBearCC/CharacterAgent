from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import scoped_session
from .models import Entity

class EntityMemory:
    def __init__(self, session: scoped_session):
        self.session = session

    def _get_entity_by_uid(self, user_guid: str):
        """Private method to retrieve a specific entity by its ID."""
        return self.session.query(Entity).filter_by(user_guid=user_guid).first()

    def add_entity(self, entity: Entity):
        """Add a new entity to the database."""
        try:
            self.session.add(entity)
            self.session.commit()
        except SQLAlchemyError as e:
            self.session.rollback()
            raise e

    def get_entities(self, user_guid: str, count: int = 100):
        """Retrieve the latest entities for a given user GUID."""
        return (self.session.query(Entity)
                .filter_by(user_guid=user_guid)
                .order_by(Entity.entity_id.desc())
                .limit(count)
                .all())

    def get_entity(self, user_guid: str):
        """Retrieve the latest entities for a given user GUID."""
        return self._get_entity_by_uid(user_guid)

    def update_entity(self, user_guid: str, entity=None, **kwargs):
        """Update specific fields of an existing entity or update using an entity object."""
        try:
            if entity is None:
                entity = self._get_entity_by_uid(user_guid)
                if not entity:
                    return None
            if kwargs:
                for key, value in kwargs.items():
                    setattr(entity, key, value)
            if entity is not None and not kwargs:
                self.session.merge(entity)

            self.session.commit()
            return entity
        except SQLAlchemyError as e:
            self.session.rollback()
            raise e


