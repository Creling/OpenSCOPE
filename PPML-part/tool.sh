#!/bin/bash

echo "" > "$1"

cd build

for i in {1..10}
do
    echo "=== $i ==="

    while true; do

        export CUDA_VISIBLE_DEVICES=""

        ./cnn_anonymous_server >> ../"$1" 2>&1 &
        server_pid=$!

        sleep 1

        ./cnn_anonymous_client > /dev/null 2>&1 &

        sleep 2

        if ! kill -0 $server_pid 2>/dev/null; then
            echo "Server crashed, restarting..." >> ../"$1"
            continue  
        fi

        wait $server_pid

        break  
    done

    pkill -f "cnn_anonymous_server_cpu" 2>/dev/null
done

grep "Total execution time" ../"$1" | awk '{sum += $4; count++} END {print "Average:", sum / count, "ms"}'