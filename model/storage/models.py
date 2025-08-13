from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Date, JSON, Integer, UniqueConstraint

Base = declarative_base()


class Models:
    class Query(Base):
        __tablename__ = "queries"
        query = Column(String, primary_key=True)


    class ProductFeatures(Base):
        __tablename__ = "product_features"
        id = Column(Integer, primary_key=True, autoincrement=True)
        query = Column(String, nullable=False, index=True)
        product_id = Column(String, nullable=False, index=True)
        data = Column(JSON, nullable=False)
        data_hash = Column(String(64), nullable=False, index=True)
        __table_args__ = (
            UniqueConstraint("query", "product_id", name="uq_product_features_qpid"),
        )

    class ProductTexts(Base):
        __tablename__ = "product_texts"
        id = Column(Integer, primary_key=True, autoincrement=True)
        query = Column(String, nullable=False, index=True)
        product_id = Column(String, nullable=False, index=True)
        data = Column(JSON, nullable=False)
        data_hash = Column(String(64), nullable=False, index=True)
        __table_args__ = (
            UniqueConstraint("query", "product_id", name="uq_product_texts_qpid"),
        )

    class ReviewFeatures(Base):
        __tablename__ = "review_features"
        id = Column(Integer, primary_key=True, autoincrement=True)
        query = Column(String, nullable=False, index=True)
        product_id = Column(String, nullable=False, index=True)
        data = Column(JSON, nullable=False)
        data_hash = Column(String(64), nullable=False, index=True)
        __table_args__ = (
            UniqueConstraint("query", "product_id", name="uq_review_features_qpid"),
        )

    class ReviewTexts(Base):
        __tablename__ = "review_texts"
        id = Column(Integer, primary_key=True, autoincrement=True)
        query = Column(String, nullable=False, index=True)
        product_id = Column(String, nullable=False, index=True)
        data = Column(JSON, nullable=False)
        data_hash = Column(String(64), nullable=False, index=True)
        __table_args__ = (
            UniqueConstraint("query", "product_id", name="uq_review_texts_qpid"),
        )