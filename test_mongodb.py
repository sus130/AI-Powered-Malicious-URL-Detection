""" In this code we are trying to check whether there is a successful connection to MongoDB or not """

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Your MongoDB Atlas URI
uri = "mongodb+srv://Jyothipriya:12345@cluster0.ubxzzwy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("✅ Pinged your deployment. You successfully connected to MongoDB Atlas!")
except Exception as e:
    print("❌ Connection failed:", e)
