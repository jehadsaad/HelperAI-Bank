from database.database import Base, engine
from database.models import Prediction


Base.metadata.create_all(bind=engine)

print("Database initialized successfully.")