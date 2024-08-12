import threading
from flask import Flask
import time
from trader2 import main
import datetime
import pytz
import asyncio

app = Flask(__name__)

@app.route('/')
def home():
    return 'Welcome to the Flask App!'


def is_within_no_trade_period(current_time, event_time):
    start_no_trade = event_time - datetime.timedelta(minutes=30)
    end_no_trade = event_time + datetime.timedelta(minutes=30)
    return start_no_trade <= current_time <= end_no_trade

def get_next_event_times(current_time):
    event_times = []

    year, month = current_time.year, current_time.month

    # NFP: First Friday of every month at 8:30 AM ET
    for day in range(1, 8):
        potential_date = datetime.date(year, month, day)
        if potential_date.weekday() == 4:  # Friday
            event_time = datetime.datetime.combine(potential_date, datetime.time(8, 30))
            event_times.append(pytz.timezone('US/Eastern').localize(event_time).astimezone(pytz.utc))
            break
    
    # FOMC: Example FOMC dates
    fomc_dates = [
        datetime.datetime(2024, 1, 31, 14, 0),
        datetime.datetime(2024, 3, 20, 14, 0),
        datetime.datetime(2024, 4, 30, 14, 0),
        datetime.datetime(2024, 6, 19, 14, 0),
        datetime.datetime(2024, 7, 31, 14, 0),
        datetime.datetime(2024, 9, 18, 14, 0),
        datetime.datetime(2024, 10, 30, 14, 0),
        datetime.datetime(2024, 12, 11, 14, 0),
    ]
    for fomc_date in fomc_dates:
        event_times.append(pytz.timezone('US/Eastern').localize(fomc_date).astimezone(pytz.utc))

    # ECB: Example ECB dates
    ecb_dates = [
        datetime.datetime(2024, 1, 11, 7, 45),
        datetime.datetime(2024, 1, 11, 8, 30),
        datetime.datetime(2024, 2, 15, 7, 45),
        datetime.datetime(2024, 2, 15, 8, 30),
        datetime.datetime(2024, 3, 14, 7, 45),
        datetime.datetime(2024, 3, 14, 8, 30),
        datetime.datetime(2024, 4, 11, 7, 45),
        datetime.datetime(2024, 4, 11, 8, 30),
        datetime.datetime(2024, 5, 9, 7, 45),
        datetime.datetime(2024, 5, 9, 8, 30),
        datetime.datetime(2024, 6, 6, 7, 45),
        datetime.datetime(2024, 6, 6, 8, 30),
        datetime.datetime(2024, 7, 25, 7, 45),
        datetime.datetime(2024, 7, 25, 8, 30),
        datetime.datetime(2024, 9, 12, 7, 45),
        datetime.datetime(2024, 9, 12, 8, 30),
        datetime.datetime(2024, 10, 24, 7, 45),
        datetime.datetime(2024, 10, 24, 8, 30),
        datetime.datetime(2024, 12, 12, 7, 45),
        datetime.datetime(2024, 12, 12, 8, 30),
    ]
    for ecb_date in ecb_dates:
        event_times.append(pytz.timezone('US/Eastern').localize(ecb_date).astimezone(pytz.utc))

    # BoE: Example BoE dates
    boe_dates = [
        datetime.datetime(2024, 2, 1, 7, 0),
        datetime.datetime(2024, 3, 21, 7, 0),
        datetime.datetime(2024, 5, 2, 7, 0),
        datetime.datetime(2024, 6, 20, 7, 0),
        datetime.datetime(2024, 8, 1, 7, 0),
        datetime.datetime(2024, 9, 19, 7, 0),
        datetime.datetime(2024, 11, 7, 7, 0),
        datetime.datetime(2024, 12, 19, 7, 0),
    ]
    for boe_date in boe_dates:
        event_times.append(pytz.timezone('US/Eastern').localize(boe_date).astimezone(pytz.utc))

    # BoJ: Example BoJ dates
    boj_dates = [
        datetime.datetime(2024, 1, 22, 0, 0),
        datetime.datetime(2024, 3, 18, 0, 0),
        datetime.datetime(2024, 4, 26, 0, 0),
        datetime.datetime(2024, 6, 20, 0, 0),
        datetime.datetime(2024, 7, 15, 0, 0),
        datetime.datetime(2024, 9, 18, 0, 0),
        datetime.datetime(2024, 10, 30, 0, 0),
        datetime.datetime(2024, 12, 18, 0, 0),
    ]
    for boj_date in boj_dates:
        event_times.append(pytz.timezone('Asia/Tokyo').localize(boj_date).astimezone(pytz.utc))

    # GDP: Example GDP dates
    gdp_dates = [
        datetime.datetime(2024, 1, 26, 8, 30),
        datetime.datetime(2024, 4, 26, 8, 30),
        datetime.datetime(2024, 7, 26, 8, 30),
        datetime.datetime(2024, 10, 26, 8, 30),
    ]
    for gdp_date in gdp_dates:
        event_times.append(pytz.timezone('US/Eastern').localize(gdp_date).astimezone(pytz.utc))

    # CPI: Example CPI dates
    cpi_dates = [
        datetime.datetime(2024, 1, 12, 8, 30),
        datetime.datetime(2024, 2, 13, 8, 30),
        datetime.datetime(2024, 3, 13, 8, 30),
        datetime.datetime(2024, 4, 10, 8, 30),
        datetime.datetime(2024, 5, 10, 8, 30),
        datetime.datetime(2024, 6, 12, 8, 30),
        datetime.datetime(2024, 7, 12, 8, 30),
        datetime.datetime(2024, 8, 12, 8, 30),
        datetime.datetime(2024, 9, 11, 8, 30),
        datetime.datetime(2024, 10, 10, 8, 30),
        datetime.datetime(2024, 11, 12, 8, 30),
        datetime.datetime(2024, 12, 12, 8, 30),
    ]
    for cpi_date in cpi_dates:
        event_times.append(pytz.timezone('US/Eastern').localize(cpi_date).astimezone(pytz.utc))

    # Retail Sales: Example Retail Sales dates
    retail_sales_dates = [
        datetime.datetime(2024, 1, 17, 8, 30),
        datetime.datetime(2024, 2, 14, 8, 30),
        datetime.datetime(2024, 3, 14, 8, 30),
        datetime.datetime(2024, 4, 16, 8, 30),
        datetime.datetime(2024, 5, 15, 8, 30),
        datetime.datetime(2024, 6, 14, 8, 30),
        datetime.datetime(2024, 7, 16, 8, 30),
        datetime.datetime(2024, 8, 15, 8, 30),
        datetime.datetime(2024, 9, 13, 8, 30),
        datetime.datetime(2024, 10, 15, 8, 30),
        datetime.datetime(2024, 11, 14, 8, 30),
        datetime.datetime(2024, 12, 13, 8, 30),
    ]
    for retail_sales_date in retail_sales_dates:
        event_times.append(pytz.timezone('US/Eastern').localize(retail_sales_date).astimezone(pytz.utc))

    return event_times


@app.route('/start-task')
def start_task():
    # Get the current server time (UTC)
    current_time = datetime.datetime.now(pytz.utc)

    # Get all event times
    event_times = get_next_event_times(current_time)

    # Check if current time is within no-trade periods for any event
    no_trade = False
    for event_time in event_times:
        if is_within_no_trade_period(current_time, event_time):
            no_trade = True
            break

    if no_trade:
        print("No trading allowed at this time.")
        return 'No trading allowed at this time.'
    else:
        print("Trading is allowed.")
        thread = threading.Thread(target=main)
        thread.start()
        return 'Task has been started in the background!'

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Specify your desired port number here
