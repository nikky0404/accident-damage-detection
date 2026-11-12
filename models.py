"""
Database models for 360° Auto Vision System.
Supports MySQL and SQLite via SQLAlchemy.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    password = Column(String(255), nullable=False, default="demo123")
    role = Column(String(50), default="inspector")
    created_at = Column(DateTime, default=datetime.utcnow)

    vehicles = relationship("Vehicle", back_populates="owner", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Vehicle(Base):
    __tablename__ = 'vehicles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    manufacturer = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False)
    year = Column(Integer, nullable=False, default=2021)
    registration_number = Column(String(50), nullable=True, default="MH-12-AB-1234")
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="vehicles")
    inspections = relationship("Inspection", back_populates="vehicle", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "year": self.year,
            "registration_number": self.registration_number,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Inspection(Base):
    __tablename__ = 'inspections'

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey('vehicles.id'), nullable=False)
    inspection_date = Column(DateTime, default=datetime.utcnow)
    image_count = Column(Integer, default=0)
    total_damage = Column(Integer, default=0)
    estimated_cost = Column(Float, default=0.0)
    cost_range_min = Column(Float, default=0.0)
    cost_range_max = Column(Float, default=0.0)
    status = Column(String(50), default="Completed")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    vehicle = relationship("Vehicle", back_populates="inspections")
    damage_records = relationship("DamageRecord", back_populates="inspection", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "vehicle_id": self.vehicle_id,
            "inspection_date": self.inspection_date.strftime("%Y-%m-%d %H:%M") if self.inspection_date else None,
            "image_count": self.image_count,
            "total_damage": self.total_damage,
            "estimated_cost": self.estimated_cost,
            "cost_range_min": self.cost_range_min,
            "cost_range_max": self.cost_range_max,
            "status": self.status,
            "notes": self.notes,
            "vehicle": self.vehicle.to_dict() if self.vehicle else None
        }

class DamageRecord(Base):
    __tablename__ = 'damage_records'

    id = Column(Integer, primary_key=True, autoincrement=True)
    inspection_id = Column(Integer, ForeignKey('inspections.id'), nullable=False)
    image_name = Column(String(255), nullable=True)
    view_name = Column(String(100), nullable=True)
    vehicle_part = Column(String(100), nullable=False)
    damage_type = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)  # Minor, Moderate, Severe
    confidence = Column(Float, default=0.85)
    damaged_area = Column(Float, default=0.0)  # Percentage (e.g. 18.7)
    repair_action = Column(String(100), default="Repair")  # Repair, Paint/Repair, Replace
    parts_cost = Column(Float, default=0.0)
    labor_cost = Column(Float, default=0.0)
    repair_cost = Column(Float, default=0.0)
    bbox_json = Column(Text, nullable=True)
    is_fused_primary = Column(Boolean, default=True)

    inspection = relationship("Inspection", back_populates="damage_records")

    def to_dict(self):
        return {
            "id": self.id,
            "inspection_id": self.inspection_id,
            "image_name": self.image_name,
            "view_name": self.view_name,
            "vehicle_part": self.vehicle_part,
            "damage_type": self.damage_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "damaged_area": self.damaged_area,
            "repair_action": self.repair_action,
            "parts_cost": self.parts_cost,
            "labor_cost": self.labor_cost,
            "repair_cost": self.repair_cost,
            "bbox_json": self.bbox_json,
            "is_fused_primary": self.is_fused_primary
        }

class RepairCostMaster(Base):
    __tablename__ = 'repair_costs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_part = Column(String(100), nullable=False)
    damage_type = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)
    parts_cost = Column(Float, default=0.0)
    labor_cost = Column(Float, default=0.0)
    base_action = Column(String(100), default="Repair")

    def to_dict(self):
        return {
            "id": self.id,
            "vehicle_part": self.vehicle_part,
            "damage_type": self.damage_type,
            "severity": self.severity,
            "parts_cost": self.parts_cost,
            "labor_cost": self.labor_cost,
            "base_action": self.base_action
        }
