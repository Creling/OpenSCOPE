#!/bin/bash

START_PORT=8800
END_PORT=8839

for ((PORT=START_PORT; PORT<=END_PORT; PORT++)); do
    CUDA_VISIBLE_DEVICES= /usr/bin/time -v ./cnn_anonymous_server "$PORT" > "server_output_$PORT.log" 2>&1 &
    echo "Server started on port $PORT (output: server_output_$PORT.log)"
done

echo "All server processes started. Waiting briefly to ensure servers are ready..."
sleep 5 

for ((PORT=START_PORT; PORT<=END_PORT; PORT++)); do
    CUDA_VISIBLE_DEVICES= ./cnn_anonymous_client "$PORT" &
    echo "Client started for port $PORT"
done

echo "All client processes started."
