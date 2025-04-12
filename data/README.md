# Getting Data

## From Postgres to CSV

1. Run the command to connect to postgres:

    ```sh
    psql "URL"
    ```

2. Download the selected data:

    ```sql
    \copy (SELECT * FROM congestion WHERE start_time >= 'start' AND start_time <= 'end') TO 'file-name.csv' WITH CSV HEADER;
    ```
