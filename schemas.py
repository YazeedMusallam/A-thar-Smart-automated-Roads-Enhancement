from pydantic import BaseModel


class PredBaseSchema(BaseModel):
    area: str
    depth: str
    no_of_potholes: str
    category: str
    status: str
    def __init__(self, area, depth, no_of_potholes, category,status):
        self.area = area
        self.depth = depth
        self.no_of_potholes = no_of_potholes
        self.category = category
        self.status = status

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
