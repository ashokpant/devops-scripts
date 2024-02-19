#!/bin/bash
SERVICE_NAME=elk
DOCKER_COMPOSE_FILE=./docker-compose.yml

start() {
    cd docker-elk
    echo "Starting $SERVICE_NAME service ..."
    docker compose -f $DOCKER_COMPOSE_FILE up -d 
    cd ..
}

stop() {
     cd docker-elk
    echo "Stopping $SERVICE_NAME service ..."
    docker compose -f $DOCKER_COMPOSE_FILE down 
    cd ..
}

# Check the command-line argument to determine action
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        start
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac

exit 0

