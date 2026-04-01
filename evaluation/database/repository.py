"""
Database repository for common operations.

Provides high-level database access methods.
"""

from datetime import datetime
from typing import Generic, List, Optional, Type, TypeVar

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from evaluation.database.models import (
    Study,
    Extraction,
    Analysis,
    AnalysisStudy,
    User,
    StudyComment,
    Job,
    Base
)

T = TypeVar('T', bound=Base)


class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations."""

    def __init__(self, model: Type[T]):
        self.model = model

    def get(self, db: Session, id: str) -> Optional[T]:
        """Get entity by ID."""
        return db.query(self.model).filter(self.model.id == id).first()

    def get_all(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100
    ) -> List[T]:
        """Get all entities with pagination."""
        return db.query(self.model).offset(skip).limit(limit).all()

    def create(self, db: Session, obj: T) -> T:
        """Create new entity."""
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj

    def update(self, db: Session, obj: T) -> T:
        """Update existing entity."""
        db.commit()
        db.refresh(obj)
        return obj

    def delete(self, db: Session, id: str) -> bool:
        """Delete entity by ID."""
        obj = self.get(db, id)
        if obj:
            db.delete(obj)
            db.commit()
            return True
        return False

    def count(self, db: Session) -> int:
        """Count all entities."""
        return db.query(func.count(self.model.id)).scalar()


class StudyRepository(BaseRepository[Study]):
    """Repository for Study entities."""

    def __init__(self):
        super().__init__(Study)

    def get_by_pmcid(self, db: Session, pmcid: str) -> Optional[Study]:
        """Get study by PMCID."""
        return db.query(Study).filter(Study.pmcid == pmcid).first()

    def get_by_pmid(self, db: Session, pmid: str) -> Optional[Study]:
        """Get study by PMID."""
        return db.query(Study).filter(Study.pmid == pmid).first()

    def search(
        self,
        db: Session,
        query: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Study]:
        """Search studies by title, authors, or keywords."""
        search_filter = f'%{query}%'
        return db.query(Study).filter(
            (Study.title.ilike(search_filter)) |
            (Study.authors.ilike(search_filter))
        ).offset(skip).limit(limit).all()

    def get_by_screening_status(
        self,
        db: Session,
        status: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Study]:
        """Get studies by screening status."""
        return db.query(Study).filter(
            Study.screening_status == status
        ).offset(skip).limit(limit).all()

    def get_by_year_range(
        self,
        db: Session,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None
    ) -> List[Study]:
        """Get studies within year range."""
        query = db.query(Study)
        if year_start:
            query = query.filter(Study.year >= year_start)
        if year_end:
            query = query.filter(Study.year <= year_end)
        return query.all()


class ExtractionRepository(BaseRepository[Extraction]):
    """Repository for Extraction entities."""

    def __init__(self):
        super().__init__(Extraction)

    def get_by_study(
        self,
        db: Session,
        study_id: str
    ) -> List[Extraction]:
        """Get all extractions for a study."""
        return db.query(Extraction).filter(
            Extraction.study_id == study_id
        ).all()

    def get_by_model(
        self,
        db: Session,
        model_name: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Extraction]:
        """Get extractions by model name."""
        return db.query(Extraction).filter(
            Extraction.model_name == model_name
        ).offset(skip).limit(limit).all()

    def get_by_outcome_type(
        self,
        db: Session,
        outcome_type: str
    ) -> List[Extraction]:
        """Get extractions by outcome type."""
        return db.query(Extraction).filter(
            Extraction.outcome_type == outcome_type
        ).all()

    def get_validated(
        self,
        db: Session,
        status: str = 'validated'
    ) -> List[Extraction]:
        """Get validated extractions."""
        return db.query(Extraction).filter(
            Extraction.validation_status == status
        ).all()

    def get_recent(
        self,
        db: Session,
        hours: int = 24,
        limit: int = 100
    ) -> List[Extraction]:
        """Get recent extractions within last N hours."""
        cutoff = datetime.utcnow()
        # Simple timestamp comparison
        return db.query(Extraction).order_by(
            Extraction.extraction_timestamp.desc()
        ).limit(limit).all()


class AnalysisRepository(BaseRepository[Analysis]):
    """Repository for Analysis entities."""

    def __init__(self):
        super().__init__(Analysis)

    def get_by_type(
        self,
        db: Session,
        analysis_type: str
    ) -> List[Analysis]:
        """Get analyses by type."""
        return db.query(Analysis).filter(
            Analysis.analysis_type == analysis_type
        ).all()

    def get_recent(
        self,
        db: Session,
        limit: int = 50
    ) -> List[Analysis]:
        """Get recent analyses."""
        return db.query(Analysis).order_by(
            Analysis.created_at.desc()
        ).limit(limit).all()

    def add_study(
        self,
        db: Session,
        analysis_id: str,
        study_id: str,
        extraction_id: Optional[str] = None,
        weight: Optional[float] = None,
        subgroup: Optional[str] = None
    ) -> AnalysisStudy:
        """Add a study to an analysis."""
        link = AnalysisStudy(
            analysis_id=analysis_id,
            study_id=study_id,
            extraction_id=extraction_id,
            weight=weight,
            subgroup=subgroup
        )
        db.add(link)
        db.commit()
        db.refresh(link)
        return link

    def remove_study(
        self,
        db: Session,
        analysis_id: str,
        study_id: str
    ) -> bool:
        """Remove a study from an analysis."""
        link = db.query(AnalysisStudy).filter(
            AnalysisStudy.analysis_id == analysis_id,
            AnalysisStudy.study_id == study_id
        ).first()
        if link:
            db.delete(link)
            db.commit()
            return True
        return False

    def get_included_studies(
        self,
        db: Session,
        analysis_id: str
    ) -> List[Study]:
        """Get all studies included in an analysis."""
        return db.query(Study).join(
            AnalysisStudy,
            Study.id == AnalysisStudy.study_id
        ).filter(
            AnalysisStudy.analysis_id == analysis_id,
            AnalysisStudy.included == True
        ).all()


class UserRepository(BaseRepository[User]):
    """Repository for User entities."""

    def __init__(self):
        super().__init__(User)

    def get_by_email(self, db: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter(User.email == email).first()

    def get_by_api_key(self, db: Session, api_key: str) -> Optional[User]:
        """Get user by API key."""
        return db.query(User).filter(User.api_key == api_key).first()

    def get_active(self, db: Session) -> List[User]:
        """Get all active users."""
        return db.query(User).filter(User.is_active == True).all()


class JobRepository(BaseRepository[Job]):
    """Repository for Job entities."""

    def __init__(self):
        super().__init__(Job)

    def get_by_status(
        self,
        db: Session,
        status: str,
        limit: int = 100
    ) -> List[Job]:
        """Get jobs by status."""
        return db.query(Job).filter(
            Job.status == status
        ).order_by(Job.created_at.asc()).limit(limit).all()

    def get_by_user(
        self,
        db: Session,
        user_id: str
    ) -> List[Job]:
        """Get jobs for a user."""
        return db.query(Job).filter(
            Job.user_id == user_id
        ).order_by(Job.created_at.desc()).all()

    def get_pending(self, db: Session) -> List[Job]:
        """Get pending jobs."""
        return self.get_by_status(db, 'pending')

    def update_status(
        self,
        db: Session,
        job_id: str,
        status: str,
        progress: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> Optional[Job]:
        """Update job status."""
        job = self.get(db, job_id)
        if job:
            job.status = status
            if progress is not None:
                job.progress = progress
            if error_message:
                job.error_message = error_message

            if status == 'running' and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status in ['completed', 'failed', 'cancelled']:
                job.completed_at = datetime.utcnow()

            db.commit()
            db.refresh(job)
        return job

    def increment_progress(
        self,
        db: Session,
        job_id: str,
        step_name: Optional[str] = None
    ) -> Optional[Job]:
        """Increment job progress."""
        job = self.get(db, job_id)
        if job and job.total_steps:
            # Simple increment
            job.progress = min(100.0, job.progress + (100.0 / job.total_steps))
            if step_name:
                job.current_step = step_name
            db.commit()
            db.refresh(job)
        return job


# Singleton instances
studies = StudyRepository()
extractions = ExtractionRepository()
analyses = AnalysisRepository()
users = UserRepository()
jobs = JobRepository()
