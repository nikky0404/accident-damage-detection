"""
Database initialization and session management.
Supports both SQLite (default local) and MySQL (production/configured).
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
try:
    from database.models import Base, User, Vehicle, Inspection, DamageRecord, RepairCostMaster
except ModuleNotFoundError:
    from models import Base, User, Vehicle, Inspection, DamageRecord, RepairCostMaster

# Default to SQLite local database file
DEFAULT_SQLITE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "accident_vision.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True
)

SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Create all tables and seed default baseline pricing & demo data if empty."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Check if master repair costs seeded
        if db.query(RepairCostMaster).count() == 0:
            seed_repair_costs(db)
        # Check if demo user & vehicle exist
        if db.query(User).count() == 0:
            seed_demo_data(db)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DB Warning] Error during init_db: {e}")
    finally:
        db.close()

def seed_repair_costs(db):
    """Seed benchmark repair & labor costs for various vehicle parts & damages in INR (₹)."""
    benchmarks = [
        # Bumper
        ("Front Bumper", "Dent", "Minor", 1200.0, 1000.0, "Repair"),
        ("Front Bumper", "Dent", "Moderate", 3000.0, 2000.0, "Repair"),
        ("Front Bumper", "Dent", "Severe", 6500.0, 3000.0, "Replace"),
        ("Front Bumper", "Scratch", "Minor", 800.0, 1200.0, "Paint/Repair"),
        ("Front Bumper", "Scratch", "Moderate", 1800.0, 1800.0, "Paint/Repair"),
        ("Front Bumper", "Crack", "Moderate", 3500.0, 2200.0, "Repair"),
        ("Front Bumper", "Broken Part", "Severe", 8500.0, 2500.0, "Replace"),
        
        ("Rear Bumper", "Dent", "Minor", 1100.0, 950.0, "Repair"),
        ("Rear Bumper", "Dent", "Moderate", 2800.0, 1800.0, "Repair"),
        ("Rear Bumper", "Dent", "Severe", 6000.0, 2800.0, "Replace"),
        ("Rear Bumper", "Scratch", "Minor", 750.0, 1100.0, "Paint/Repair"),
        
        # Hood
        ("Hood", "Dent", "Minor", 1500.0, 1500.0, "Repair"),
        ("Hood", "Dent", "Moderate", 4000.0, 2500.0, "Repair"),
        ("Hood", "Dent", "Severe", 11000.0, 3500.0, "Replace"),
        ("Hood", "Scratch", "Minor", 1000.0, 1800.0, "Paint/Repair"),
        ("Hood", "Scratch", "Moderate", 2500.0, 2500.0, "Paint/Repair"),
        ("Hood", "Deformation", "Severe", 12500.0, 4000.0, "Replace"),

        # Doors & Fenders
        ("Door", "Dent", "Minor", 1400.0, 1400.0, "Repair"),
        ("Door", "Dent", "Moderate", 3800.0, 2200.0, "Repair"),
        ("Door", "Dent", "Severe", 9500.0, 3000.0, "Replace"),
        ("Door", "Scratch", "Minor", 900.0, 1600.0, "Paint/Repair"),
        ("Fender", "Dent", "Minor", 1200.0, 1200.0, "Repair"),
        ("Fender", "Dent", "Moderate", 3000.0, 1800.0, "Repair"),
        ("Fender", "Scratch", "Minor", 800.0, 1400.0, "Paint/Repair"),
        ("Fender", "Deformation", "Severe", 5500.0, 2500.0, "Replace"),

        # Lights & Glass
        ("Headlight", "Crack", "Moderate", 4500.0, 800.0, "Replace"),
        ("Headlight", "Crack", "Severe", 8000.0, 1000.0, "Replace"),
        ("Headlight", "Broken Part", "Severe", 8500.0, 1200.0, "Replace"),
        ("Tail light", "Crack", "Moderate", 3200.0, 600.0, "Replace"),
        ("Tail light", "Broken Part", "Severe", 5500.0, 800.0, "Replace"),
        ("Windshield", "Crack", "Moderate", 4500.0, 1500.0, "Repair/Seal"),
        ("Windshield", "Crack", "Severe", 9500.0, 2500.0, "Replace"),
        ("Side Mirror", "Broken Part", "Severe", 3800.0, 700.0, "Replace"),
        ("Side Mirror", "Scratch", "Minor", 600.0, 500.0, "Paint/Repair"),
        ("Trunk", "Dent", "Moderate", 3500.0, 2000.0, "Repair"),
        ("Roof", "Dent", "Moderate", 4200.0, 2800.0, "Repair")
    ]
    for part, dtype, sev, p_cost, l_cost, action in benchmarks:
        record = RepairCostMaster(
            vehicle_part=part,
            damage_type=dtype,
            severity=sev,
            parts_cost=p_cost,
            labor_cost=l_cost,
            base_action=action
        )
        db.add(record)

def seed_demo_data(db):
    """Seed demo user, vehicle and an initial inspection record."""
    demo_user = User(
        name="Auto Vision Inspector",
        email="inspector@autovision360.ai",
        password="admin",
        role="senior_inspector"
    )
    db.add(demo_user)
    db.flush()

    demo_veh1 = Vehicle(
        user_id=demo_user.id,
        manufacturer="Hyundai",
        model="Creta SX",
        year=2023,
        registration_number="DL-01-AX-4821"
    )
    demo_veh2 = Vehicle(
        user_id=demo_user.id,
        manufacturer="Tata",
        model="Nexon EV",
        year=2022,
        registration_number="MH-02-CP-9012"
    )
    db.add(demo_veh1)
    db.add(demo_veh2)
    db.flush()

    # Pre-seed a sample completed inspection
    insp = Inspection(
        vehicle_id=demo_veh1.id,
        image_count=4,
        total_damage=3,
        estimated_cost=19500.0,
        cost_range_min=17000.0,
        cost_range_max=22000.0,
        status="Completed",
        notes="Front collision with barrier; bumper dent, left fender scratch, and cracked right headlight."
    )
    db.add(insp)
    db.flush()

    damages = [
        DamageRecord(
            inspection_id=insp.id,
            image_name="front_view.jpg",
            view_name="Front View",
            vehicle_part="Front Bumper",
            damage_type="Dent",
            severity="Moderate",
            confidence=0.94,
            damaged_area=18.7,
            repair_action="Repair",
            parts_cost=4000.0,
            labor_cost=2000.0,
            repair_cost=6000.0,
            bbox_json='{"x": 120, "y": 280, "w": 260, "h": 140}',
            is_fused_primary=True
        ),
        DamageRecord(
            inspection_id=insp.id,
            image_name="front_left_view.jpg",
            view_name="Front-Left View",
            vehicle_part="Left Fender",
            damage_type="Scratch",
            severity="Minor",
            confidence=0.91,
            damaged_area=8.4,
            repair_action="Paint/Repair",
            parts_cost=3000.0,
            labor_cost=1500.0,
            repair_cost=4500.0,
            bbox_json='{"x": 50, "y": 210, "w": 180, "h": 90}',
            is_fused_primary=True
        ),
        DamageRecord(
            inspection_id=insp.id,
            image_name="front_view.jpg",
            view_name="Front View",
            vehicle_part="Headlight",
            damage_type="Crack",
            severity="Severe",
            confidence=0.92,
            damaged_area=24.2,
            repair_action="Replace",
            parts_cost=8000.0,
            labor_cost=1000.0,
            repair_cost=9000.0,
            bbox_json='{"x": 340, "y": 190, "w": 110, "h": 85}',
            is_fused_primary=True
        )
    ]
    for d in damages:
        db.add(d)
