# Self Coaching System Using ARDUINO UNO and MPU6050 along with NRF24L01.
A self-coaching system using an arduino uno microprocessor along with a NRF24L01 sensor to provide communication between the "sender" and "receiver" arduino modules.
6 Different components of the produced data ( accX, accY, accZ, gyroX, gyroY, gyroZ) is transmitted from sender to receiver module and written to the cloud ( local computer ) via serial connection. 

Collected data is preprocessed using kalman-filtering, central tendency and distribution measurements. 
A set of models were generated :a neural network in pytorch and diverse machine learning and ensemble algorithms:

Random forest performed best for the case. However, given example data is not qualified and not produced by professionals.
Collecting a better data can lead to a better performing neural-network model.

Proposed codes can be used for different movement-acceleration based calculations as well.
