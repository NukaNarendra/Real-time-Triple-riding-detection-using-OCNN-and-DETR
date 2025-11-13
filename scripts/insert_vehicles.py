import sys
sys.path.insert(0, ".")
from db.session import SessionLocal
from db.models import RegisteredVehicle

vehicles = [
    ("AP39AB1234", "+919876543210"),
    ("TS09ZX9876", "+919998887776"),
    ("AP10CD4567", "+919112223334"),
    ("TS07MK4321", "+919445566778"),
    ("AP31XY1111", "+917770001234"),
    ("TS08LM2222", "+917770002345"),
    ("AP05GH3333", "+917770003456"),
    ("TS06JK4444", "+917770004567"),
    ("AP21PQ5555", "+917770005678"),
    ("TS12RS6666", "+917770006789"),
    ("AP13TU7777", "+917770007890"),
    ("TS14VW8888", "+917770008901"),
    ("AP15YZ9999", "+917770009012"),
    ("TS02AB1010", "+917770000123"),
    ("AP03NM2020", "+917770000456"),
    ("TS04GH3030", "+917770000789"),
    ("AP07IP4040", "+917770000321"),
    ("TS11JK5050", "+917770000654"),
    ("AP18LM6060", "+917770000987"),
    ("TS20QR7070", "+917770000852"),
]

session = SessionLocal()
inserted = 0
for plate, phone in vehicles:
    # avoid duplicates
    exists = session.query(RegisteredVehicle).filter(RegisteredVehicle.plate_text == plate).first()
    if not exists:
        v = RegisteredVehicle(plate_text=plate, phone_number=phone)
        session.add(v)
        inserted += 1

session.commit()
session.close()
print(f"Inserted {inserted} new RegisteredVehicle rows (skipped existing).")
