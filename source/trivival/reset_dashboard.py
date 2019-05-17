from database import DashboardDatabase

db = DashboardDatabase()
db.mongodb_db['face_id'].remove({})
db.mongodb_db['changes'].remove({})
db.mongodb_db['visithistory'].remove({})
db.mongodb_db['visitor'].remove({})
db.mongodb_db['visitorold'].remove({})
db.mongodb_db['visitorstatistics'].remove({})
db.mongodb_db['visitorweeklystatistics'].remove({})
db.mongodb_db['dashinfo'].remove({})
db.mongodb_db['faceinfo'].remove({})
db.mongodb_db['mslog'].remove({})
