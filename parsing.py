import requests
from datetime import datetime, timedelta
import csv

# fetch data from Pyth Network API
def fetch_data(publish_time, price_id):
    url = f"https://hermes.pyth.network/v2/updates/price/{publish_time}?ids%5B%5D={price_id}"
    response = requests.get(url)
    return response.json()

# parse the JSON response
def parse_data(data, i):
    parsed_data = data['parsed'][0]
    price = float(parsed_data['price']['price']) * (10 ** parsed_data['price']['expo'])
    conf = float(parsed_data['price']['conf']) * (10 ** parsed_data['price']['expo'])
    ema_price = float(parsed_data['ema_price']['price']) * (10 ** parsed_data['ema_price']['expo'])
    ema_conf = float(parsed_data['ema_price']['conf']) * (10 ** parsed_data['ema_price']['expo'])
    timestamp = datetime.utcnow() - timedelta(minutes=15 * i)
    return {
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'price': price,
        'conf': conf,
        'ema_price': ema_price,
        'ema_conf': ema_conf
    }

# main function to collect data and save to CSV
def main():
    price_id = '0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace'
    latest_data = fetch_data('latest', price_id)
    latest_publish_time = latest_data['parsed'][0]['price']['publish_time']

    # Open the CSV file in write mode and write the header
    with open('eth_price_hist.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['timestamp', 'price', 'conf', 'ema_price', 'ema_conf'])
        writer.writeheader()

        for i in range(10000):
            publish_time = latest_publish_time - (i * 900)
            data = fetch_data(publish_time, price_id)
            parsed_data = parse_data(data, i)
            writer.writerow(parsed_data)  # Write each parsed data entry to the CSV file

            if i % 10 == 0:
                print(f'Collected data for {i} timestamps')

if __name__ == '__main__':
    main()