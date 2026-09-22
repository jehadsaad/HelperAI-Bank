from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from database.database import Base


class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    # Customer information
    limit_bal = Column(Float)
    sex = Column(Integer)
    education = Column(Integer)
    marriage = Column(Integer)
    age = Column(Integer)

    # Repayment status
    pay_0 = Column(Integer)
    pay_2 = Column(Integer)
    pay_3 = Column(Integer)
    pay_4 = Column(Integer)
    pay_5 = Column(Integer)
    pay_6 = Column(Integer)

    # Bill amounts
    bill_amt1 = Column(Float)
    bill_amt2 = Column(Float)
    bill_amt3 = Column(Float)
    bill_amt4 = Column(Float)
    bill_amt5 = Column(Float)
    bill_amt6 = Column(Float)

    # Payment amounts
    pay_amt1 = Column(Float)
    pay_amt2 = Column(Float)
    pay_amt3 = Column(Float)
    pay_amt4 = Column(Float)
    pay_amt5 = Column(Float)
    pay_amt6 = Column(Float)

    # Prediction result
    prediction = Column(Integer)
    default_probability = Column(Float)

    # SHAP explanation
    reasons = Column(String)

    # Timestamp
    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )