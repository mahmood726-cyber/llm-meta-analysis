"""
Database models for LLM Meta-Analysis Framework.

SQLAlchemy models for studies, extractions, analyses, and users.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Study(Base):
    """
    Represents a clinical trial study.

    Stores metadata and document information for RCTs.
    """
    __tablename__ = 'studies'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    pmcid: Mapped[Optional[str]] = mapped_column(String(50), unique=True, nullable=True, index=True)
    pmid: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    doi: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Study identifiers
    cochrane_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    clinical_trials_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Citation information
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    authors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    journal: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Document storage
    document_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    document_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # pdf, xml, markdown

    # Screening status
    screening_status: Mapped[Optional[str]] = mapped_column(
        Enum('not_screened', 'included', 'excluded', 'maybe', name='screening_status'),
        default='not_screened'
    )
    exclusion_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Quality assessment
    risk_of_bias: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    extractions: Mapped[list["Extraction"]] = relationship(
        "Extraction", back_populates="study", cascade="all, delete-orphan"
    )
    comments: Mapped[list["StudyComment"]] = relationship(
        "StudyComment", back_populates="study", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index('idx_studies_pmid', 'pmid'),
        Index('idx_studies_year', 'year'),
        Index('idx_studies_screening', 'screening_status'),
    )


class Extraction(Base):
    """
    Represents data extracted from a study.

    Stores extracted outcomes (binary, continuous, survival, etc.)
    """
    __tablename__ = 'extractions'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    study_id: Mapped[str] = mapped_column(String(36), ForeignKey('studies.id', ondelete='CASCADE'), nullable=False, index=True)

    # Model information
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., gpt-4, claude-opus
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Outcome information
    outcome_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    outcome_type: Mapped[Optional[str]] = mapped_column(
        Enum('binary', 'continuous', 'survival', 'count', 'unknown', name='outcome_type'),
        nullable=True
    )

    # Intervention/comparator
    intervention: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    comparator: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Extracted data (structured)
    data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Example for binary outcomes:
    # {
    #     "intervention_events": 10,
    #     "intervention_total": 100,
    #     "comparator_events": 15,
    #     "comparator_total": 100
    # }

    # Example for continuous outcomes:
    # {
    #     "intervention_mean": 12.5,
    #     "intervention_sd": 3.2,
    #     "intervention_n": 100,
    #     "comparator_mean": 15.0,
    #     "comparator_sd": 3.5,
    #     "comparator_n": 100
    # }

    # Example for survival outcomes:
    # {
    #     "hazard_ratio": 0.75,
    #     "hr_ci_lower": 0.60,
    #     "hr_ci_upper": 0.94,
    #     "hr_p_value": 0.012,
    #     "time_point": "12 months"
    # }

    # Extraction metadata
    extraction_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extraction_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Confidence and validation
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0-1
    validation_status: Mapped[Optional[str]] = mapped_column(
        Enum('pending', 'validated', 'corrected', 'rejected', name='validation_status'),
        default='pending'
    )
    validator_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Tokens and cost tracking
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    study: Mapped["Study"] = relationship("Study", back_populates="extractions")

    __table_args__ = (
        Index('idx_extractions_study', 'study_id'),
        Index('idx_extractions_model', 'model_name'),
        Index('idx_extractions_outcome', 'outcome_type'),
        Index('idx_extractions_timestamp', 'extraction_timestamp'),
    )


class Analysis(Base):
    """
    Represents a meta-analysis.

    Stores analysis configuration and results.
    """
    __tablename__ = 'analyses'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))

    # Analysis metadata
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    analysis_type: Mapped[str] = mapped_column(
        Enum('pairwise', 'network', 'ipd', 'cumulative', 'meta_regression', name='analysis_type'),
        nullable=False
    )

    # Analysis configuration
    outcome_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    effect_measure: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # OR, MD, SMD, HR, etc.
    model_type: Mapped[Optional[str]] = mapped_column(
        Enum('fixed', 'random', 'quality_effects', name='model_type'),
        nullable=True
    )
    tau2_method: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # DL, REML, SJ, PM
    ci_method: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # wald, hksj, quantile

    # Configuration (JSONB for flexibility)
    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Analysis results
    results: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Example for pairwise analysis:
    # {
    #     "pooled_effect": -0.5,
    #     "ci_lower": -0.8,
    #     "ci_upper": -0.2,
    #     "p_value": 0.001,
    #     "i_squared": 65.5,
    #     "tau_squared": 0.15,
    #     "n_studies": 10
    # }

    # Plots and reports
    plot_paths: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    report_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    included_studies: Mapped[list["AnalysisStudy"]] = relationship(
        "AnalysisStudy", back_populates="analysis", cascade="all, delete-orphan"
    )


class AnalysisStudy(Base):
    """
    Junction table for analyses and studies.

    Tracks which studies are included in which analyses,
    with optional exclusions and weights.
    """
    __tablename__ = 'analysis_studies'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    analysis_id: Mapped[str] = mapped_column(String(36), ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True)
    study_id: Mapped[str] = mapped_column(String(36), ForeignKey('studies.id', ondelete='CASCADE'), nullable=False, index=True)
    extraction_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey('extractions.id', ondelete='SET NULL'), nullable=True)

    # Inclusion/exclusion
    included: Mapped[bool] = mapped_column(Boolean, default=True)
    exclusion_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Analysis-specific data
    weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # In meta-analysis
    subgroup: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # For subgroup analysis

    # Relationships
    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="included_studies")

    __table_args__ = (
        UniqueConstraint('analysis_id', 'study_id', name='unique_analysis_study'),
        Index('idx_analysis_studies_analysis', 'analysis_id'),
        Index('idx_analysis_studies_study', 'study_id'),
    )


class User(Base):
    """
    Represents a user of the system.

    For authentication, authorization, and collaboration.
    """
    __tablename__ = 'users'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))

    # Authentication
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    api_key: Mapped[Optional[str]] = mapped_column(String(100), unique=True, nullable=True)

    # Profile
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    affiliation: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Role and permissions
    role: Mapped[str] = mapped_column(
        Enum('viewer', 'editor', 'admin', name='user_role'),
        default='viewer'
    )

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    comments: Mapped[list["StudyComment"]] = relationship("StudyComment", back_populates="author")


class StudyComment(Base):
    """
    Comments on studies for collaboration.

    Allows users to discuss studies, flag issues, etc.
    """
    __tablename__ = 'study_comments'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    study_id: Mapped[str] = mapped_column(String(36), ForeignKey('studies.id', ondelete='CASCADE'), nullable=False, index=True)
    author_id: Mapped[str] = mapped_column(String(36), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)

    # Comment content
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Comment metadata
    parent_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey('study_comments.id', ondelete='CASCADE'), nullable=True)  # For threaded comments
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    study: Mapped["Study"] = relationship("Study", back_populates="comments")
    author: Mapped["User"] = relationship("User", back_populates="comments")
    replies: Mapped[list["StudyComment"]] = relationship(
        "StudyComment",
        backref=backref('parent', remote_side=[id]),
        cascade="all, delete-orphan"
    )


class Job(Base):
    """
    Represents an async job (extraction, analysis, etc.).

    Tracks background jobs with status and progress.
    """
    __tablename__ = 'jobs'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))

    # Job information
    job_type: Mapped[str] = mapped_column(
        Enum('extraction', 'analysis', 'fine_tuning', 'export', name='job_type'),
        nullable=False
    )
    status: Mapped[str] = mapped_column(
        Enum('pending', 'running', 'completed', 'failed', 'cancelled', name='job_status'),
        default='pending'
    )

    # Job configuration
    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Job results
    result: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Progress tracking
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0-100
    current_step: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    total_steps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # User who created the job
    user_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index('idx_jobs_status', 'status'),
        Index('idx_jobs_type', 'job_type'),
        Index('idx_jobs_user', 'user_id'),
        Index('idx_jobs_created', 'created_at'),
    )
