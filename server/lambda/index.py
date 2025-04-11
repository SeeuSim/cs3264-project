import requests
import psycopg2

import os
from datetime import datetime, time
from zoneinfo import ZoneInfo
import json

def is_singapore_time_between_5am_and_1am():
    now = datetime.now(ZoneInfo("Asia/Singapore"))
    current_time = now.time()

    start = time(5, 0)  # 5:00 AM
    end = time(1, 0)  # 1:00 AM (next day)

    # Since time(1, 0) is less than time(5, 0), we need to wrap over midnight
    return current_time >= start or current_time < end

def lambda_handler(event, context):
    if not is_singapore_time_between_5am_and_1am():
        return {"statusCode": 200,"body": "MRT Not in Operation"}
    
    api_url = "https://datamall2.mytransport.sg/ltaodataservice/PCDRealTime"

    line_codes = [
        "CCL",
        "CEL",
        "CGL",
        "DTL",
        "EWL",
        "NEL",
        "NSL",
        "BPL",
        "SLRT",
        "PLRT",
        "TEL",
    ]

    conn = psycopg2.connect(
        host=os.environ['DB_HOST'],
        database=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        port=os.environ.get('DB_PORT')
    )

    with conn:
        with conn.cursor() as cur:
            for line in line_codes:
                process_line(api_url, line, cur)

    conn.close()
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

def process_line(url: str, line_code: str, cur: any):
    try:
        response = requests.get(
            url,
            params={"TrainLine": line_code},
            headers={
                "AccountKey": os.environ.get("LTA_ACCT_KEY"),
                "accept": "application/json",
            },
        )
        response.raise_for_status()

        json_data = response.json()
        if not json_data["value"]:
            raise Exception("Unexpected shape: " + json.dumps(json_data))

        rows = json_data["value"]

        for i, row in enumerate(rows):
            if (
                not row["Station"]
                or not row["StartTime"]
                or not row["EndTime"]
                or not row["CrowdLevel"]
            ):
                raise Exception(f"Unexpected shape at idx: {i}" + json.dumps(json_data))
            
            cur.execute("""
                INSERT INTO congestion (station, start_time, end_time, crowd_level, created_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (station, start_time, end_time)
                DO UPDATE SET
                  crowd_level = EXCLUDED.crowd_level,
                  created_at = now();
            """, (row['Station'], row['StartTime'], row['EndTime'], row['CrowdLevel']))

        print(f"Success: {line_code}")
    except requests.RequestException as e:
        print(f"Error: {line_code} " + str(e))

        cur.execute("""
            INSERT INTO logs (params, log)
            VALUES (%s, %s);
        """, (json.dumps({"line_code": line_code, "time": str(datetime.now(ZoneInfo("Asia/Singapore")))}), str(e)))

        # log_entry = Log(
        #     params=json.dumps(
        #         {
        #             "line_code": line_code,
        #             "time": str(datetime.now(ZoneInfo("Asia/Singapore"))),
        #         }
        #     ),
        #     log=str(e),
        # )
