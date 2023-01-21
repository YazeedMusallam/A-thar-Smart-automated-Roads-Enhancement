from database import Base
from sqlalchemy import Integer, Column, String, Float, Text
from sqlalchemy.sql import func
from fastapi_utils.guid_type import GUID, GUID_DEFAULT_SQLITE


class Pred(Base):
    __tablename__ = 'pred'
    id = Column(Integer, primary_key=True)
    area = Column(Text, nullable=False)
    depth = Column(Text, nullable=False)
    no_of_potholes = Column(Text, nullable=False)
    category = Column(Text, nullable=False)
    status = Column(Text, nullable=False)
