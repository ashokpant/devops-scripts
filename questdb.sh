#!/bin/bash
SERVICE_NAME=questdb
DOCKER_COMPOSE_FILE=./questdb-docker-compose.yml

start() {
    echo "Starting $SERVICE_NAME service ..."
    docker compose -f $DOCKER_COMPOSE_FILE up -d 
}

stop() {
    echo "Stopping $SERVICE_NAME service ..."
    docker compose -f $DOCKER_COMPOSE_FILE down 
}

# Check the command-line argument to determine action
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    pull)
        echo "Pulling latest images for $SERVICE_NAME service ..."
        docker compose -f $DOCKER_COMPOSE_FILE pull
        ;;
    restart)
        stop
        start
        ;;
    *)
        echo "Usage: $0 {start|stop|pull|restart}"
        exit 1
        ;;
esac

exit 0

