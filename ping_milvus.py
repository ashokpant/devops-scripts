from pymilvus import connections
connection = connections.connect(alias="default", host="127.0.0.1", port=19530)
print(connection)
