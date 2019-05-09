import pymongo

def getDBRef(password, name):
    client = pymongo.MongoClient("mongodb://root:" + password + "@npxcluster-shard-00-00-kcytj.azure.mongodb.net:27017,npxcluster-shard-00-01-kcytj.azure.mongodb.net:27017,npxcluster-shard-00-02-kcytj.azure.mongodb.net:27017/test?ssl=true&replicaSet=NPXCluster-shard-0&authSource=admin&retryWrites=true")
    return client[name]

def insertDoc(db, colName, doc):
    return db[colName].insert_one(doc)

def queryCol(db, colName):
    return db[colName].find()