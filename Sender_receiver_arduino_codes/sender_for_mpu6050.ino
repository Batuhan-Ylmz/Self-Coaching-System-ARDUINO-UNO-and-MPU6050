#include <Wire.h>
#include <MPU6050.h>
#include <RF24.h>

MPU6050 mpu;
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
  Wire.begin();
  mpu.initialize();

  radio.begin();
  radio.openWritingPipe(0xF0F0F0F0E1LL); // Set the address for the communication channel
}

void loop() {
  /*
  ALTERNATIVELY FOLLOWING DATA PACKAGES CAN BE SENT. 
  IT WILL PERFORM MORE THAN 60 - 65 % MEMORY.
  HOWEVER, IT WİLL REDUCE THE DIMENSION A LOT, WHCH COULD POTENTIALLY LEAD TO A POOR-PERFORMING MODEL
  float accResultant = sqrt(pow(accX, 2) + pow(accY, 2) + pow(accZ, 2));
  float gyroResultant = sqrt(pow(gyroX, 2) + pow(gyroY, 2) + pow(gyroZ, 2));
  */
  // Read accelerometer and gyroscope data
  // RATHER THAN FLOAT (4 bytes), TAKING THE DATA AS INT_16 (2 bytes) WHICH WILL SAVE
  // UP TO 50% MEMORY FOR THE PACKAGE SIZE.
  int16_t accX, accY, accZ, gyroX, gyroY, gyroZ;
  mpu.getMotion6(&accX, &accY, &accZ, &gyroX, &gyroY, &gyroZ);
  
  // Create a data packet as a struct
  SensorData data;
  data.accX = accX;
  data.accY = accY;
  data.accZ = accZ;
  data.gyroX = gyroX;
  data.gyroY = gyroY;
  data.gyroZ = gyroZ;

  // Send data packet
  radio.write(&data, sizeof(data));

  delay(1000);
}
