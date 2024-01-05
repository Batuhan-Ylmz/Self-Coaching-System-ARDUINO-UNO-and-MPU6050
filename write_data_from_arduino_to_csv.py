import serial
import csv
from datetime import datetime

# Set the serial port and baud rate
arduino_port = 'COM11'
baud_rate = 9600

# Open the serial connection
ser = serial.Serial(arduino_port, baud_rate, timeout=1)

# Create a CSV file and write header
csv_file_name = 'sensor_data.csv'
with open(csv_file_name, mode='w', newline='') as csv_file:
    fieldnames = ['Timestamp', 'X-axis', 'Y-axis', 'Z-axis']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    try:
        while True:
            # Read a line of data from Arduino
            data = ser.readline().decode().strip()

            # Check if the data is not empty and contains commas
            if data and ',' in data:
                # Extract numeric values directly without splitting
                x, y, z = map(int, data.split(','))

                # Get the current timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Write data to CSV file
                writer.writerow({'Timestamp': timestamp, 'X-axis': x, 'Y-axis': y, 'Z-axis': z})

                # Print the data to console (optional)
                print(f'Timestamp: {timestamp}, X-axis: {x}, Y-axis: {y}, Z-axis: {z}')

    except KeyboardInterrupt:
        # Close the serial connection on Ctrl+C
        ser.close()
