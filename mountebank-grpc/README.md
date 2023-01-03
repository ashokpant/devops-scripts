# Example
Imposter example
```json
{
    "protocol": "grpc",
    "port": 4545,
    "loglevel": "debug",
    "recordRequests": true,
    "services": {
        "example.ExampleService": {
            "file": "example.proto"
        }
    },
    "options": {
        "protobufjs": {
            "includeDirs": ["/home/ashok/Projects/scripts/mountebank-grpc/mountebank-grpc/src/protos/"]
        }
    },
    "stubs": [{
        "predicates": [
            {
                "matches": { "path": "UnaryUnary" },
                "caseSensitive": false
            }
        ],
        "responses": [
            {
                "is": {
                    "value": {
                        "id": 100,
                        "data": "mock response"
                    },
                    "metadata": {
                        "initial": {
                            "metadata-initial-key": "metadata-initial-value"
                        },
                        "trailing": {
                            "metadata-trailing-key": "metadata-trailing-value"
                        }
                    },
                    "error": {
                        "status": "OUT_OF_RANGE",
                        "message": "invalid message"
                    }
                }
            }
        ]
    }]
}
```

Curl to mountebank
curl -i -X POST -H 'Content-Type: application/json' http://localhost:2525/imposters --data '{
    "protocol": "grpc",
    "port": 4545,
    "loglevel": "debug",
    "recordRequests": true,
    "services": {
        "example.ExampleService": {
            "file": "example.proto"
        }
    },
    "options": {
        "protobufjs": {
            "includeDirs": ["/home/ashok/Projects/scripts/mountebank-grpc/mountebank-grpc/src/protos/"]
        }
    },
    "stubs": [{
        "predicates": [
            {
                "matches": { "path": "UnaryUnary" },
                "caseSensitive": false
            }
        ],
        "responses": [
            {
                "is": {
                    "value": {
                        "id": 100,
                        "data": "mock response"
                    },
                    "metadata": {
                        "initial": {
                            "metadata-initial-key": "metadata-initial-value"
                        },
                        "trailing": {
                            "metadata-trailing-key": "metadata-trailing-value"
                        }
                    },
                    "error": {
                        "status": "OUT_OF_RANGE",
                        "message": "invalid message"
                    }
                }
            }
        ]
    }]
}
'


