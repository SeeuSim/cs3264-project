from flask import Flask, jsonify
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import insert
import requests
import os
from datetime import datetime
import json

app = Flask(__name__)

db_pass = os.environ.get('DB_PASSWORD')
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://postgres.jdwvfnnqmdxostvzljbl:{db_pass}@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model for the 'congestion' table
class Congestion(db.Model):
    __tablename__ = 'congestion'
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=func.now())
    station = db.Column(db.Text, nullable=False)
    start_time = db.Column(db.Text, nullable=False)
    end_time = db.Column(db.Text, nullable=False)
    crowd_level = db.Column(db.Text, nullable=False)

    __table_args__ = (
        UniqueConstraint('station', 'start_time', 'end_time', name='uq_station_time'),
    )

# Model for the 'logs' table
class Log(db.Model):
    __tablename__ = 'logs'
    id = db.Column(db.Integer, primary_key=True)
    params = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=func.now())
    log = db.Column(db.Text, nullable=False)

migrate = Migrate(app, db)

@app.route('/')
def get_data():
    api_url = 'https://datamall2.mytransport.sg/ltaodataservice/PCDRealTime'

    line_codes = [
        'CCL',
        'CEL',
        'CGL',
        'DTL',
        'EWL',
        'NEL',
        'NSL',
        'BPL',
        'SLRT',
        'PLRT',
        'TEL',
    ]

    for line in line_codes:
        process_line(api_url, line)

    return {'message': 'Hello World'}

def process_line(url: str, line_code: str):
    try:
        response = requests.get(url, params={
            'TrainLine': line_code
        }, headers={
            'AccountKey': os.environ.get('LTA_ACCT_KEY'),
            'accept': 'application/json'
        })
        response.raise_for_status()

        json_data = response.json()
        if not json_data['value']:
            raise Exception('Unexpected shape: ' + json.dumps(json_data))
        
        rows = json_data['value']
    
        for i, row in enumerate(rows):
            if not row['Station'] or not row['StartTime'] or not row['EndTime'] or not row['CrowdLevel']:
                raise Exception(f'Unexpected shape at idx: {i}' + json.dumps(json_data))
            insert_stmt = insert(Congestion).values(
                station=row['Station'],
                start_time = row['StartTime'],
                end_time = row['EndTime'],
                crowd_level=row['CrowdLevel']
            )
            do_update_stmt = insert_stmt.on_conflict_do_update(
                index_elements=['station', 'start_time', 'end_time'],
                set_={
                    'crowd_level': insert_stmt.excluded.crowd_level,
                    'created_at': func.now()
                }
            )            
            db.session.execute(do_update_stmt)
        print(f'Success: {line_code}')
    except requests.RequestException as e:
        print(f'Error: {line_code} ' + str(e))
        log_entry = Log(
            params=json.dumps({
                'line_code': line_code,
                'time': str(datetime.now())
            }),
            log=str(e)
        )
        db.session.add(log_entry)
    db.session.commit()



if __name__ == '__main__':
    app.run(debug=True)