#include <RF24.h>

RF24 radio(9, 10); // CE, CSN

struct SensorData {
  int16_t accX;
  int16_t accY;
  int16_t accZ;
  int16_t gyroX;
  int16_t gyroY;
  int16_t gyroZ;
};

void setup() {
  Serial.begin(9600);

  radio.begin();
  radio.openReadingPipe(1, 0xF0F0F0F0E1LL); // Set the address for the communication channel
  radio.startListening();
}

void loop() {
  if (radio.available()) {
    // Read the data packet as a struct
    SensorData data;
    radio.read(&data, sizeof(data));

    // Process the received data
    int16_t accX = data.accX;
    int16_t accY = data.accY;
    int16_t accZ = data.accZ;
    int16_t gyroX = data.gyroX;
    int16_t gyroY = data.gyroY;
    int16_t gyroZ = data.gyroZ;

    // Now you can use accX, accY, accZ, gyroX, gyroY, gyroZ as needed
    Serial.print("AccX: "); Serial.println(accX);
    Serial.print("AccY: "); Serial.println(accY);
    Serial.print("AccZ: "); Serial.println(accZ);
    Serial.print("GyroX: "); Serial.println(gyroX);
    Serial.print("GyroY: "); Serial.println(gyroY);
    Serial.print("GyroZ: "); Serial.println(gyroZ);

    delay(1000);
  }
}
