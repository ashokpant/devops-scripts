#!/bin/bash
SERVICE_NAME=pinot
DOCKER_COMPOSE_FILE=./pinot-docker-compose.yml

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

