from database import AnnotationDatabase

db = AnnotationDatabase()
db.mongodb_image.remove({})
