#!/bin/bash
database=$1
wait_time=15s
password=$2
data_dir=$3

# wait for SQL Server to come up
echo importing data will start in $wait_time...
sleep $wait_time
echo importing data...

# run the init script to create the DB and the tables in /table
/opt/mssql-tools/bin/sqlcmd -S 0.0.0.0 -U sa -P $password -i ./init.sql

for entry in $data_dir"/*.sql"
do
  echo executing $entry
  /opt/mssql-tools/bin/sqlcmd -S 0.0.0.0 -U sa -P $password -i $entry
done

