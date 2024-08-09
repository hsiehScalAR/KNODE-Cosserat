// #include "HX711.h"
#include "TensionMotor.h"
// #include "AnalogLoadCell.h"

#define NUM_MOTORS 4
#define N_PRINT 10 // Print every N iterations. Printing takes time. less printing = faster frequency

#define MAX_TENSION 2300 // After this tension (g), go into emergency stop and relax tension on all motors

// TOM
#define KP 0.1512 * 3
// #define KI 0.04536
// #define KD 0.003024
#define KD 0.001 //0.01

// #define KP 0.18144
// #define KI 0.4 //03024
// #define KD 0.002 // 0.001512

// #define KP 0.09
#define KI 0.005 //03024
// #define KD 0.001 // 0.001512

// #define SETPOINT 300 // 278

// HX711 scales[NUM_MOTORS];

TensionMotor motors[] = {
  TensionMotor(24, 26, 44),
  TensionMotor(4, 5, 6),
  TensionMotor(11, 12, 13),
  TensionMotor(7, 8, 9),
};

AnalogLoadCell loadCells[] = {
  AnalogLoadCell(A4),
  AnalogLoadCell(A5),
  AnalogLoadCell(A6),
  AnalogLoadCell(A7),
};

float setpoints[NUM_MOTORS];
float readings[NUM_MOTORS];
float previousErrors[NUM_MOTORS];
float integratedErrors[NUM_MOTORS];

float tensions[NUM_MOTORS];
unsigned long currentTime, previousTime;
unsigned long counter = 0;
float accumDt = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("HX711 scale demo");

  // scales[0].begin(A1, A0); // First is DOUT
  // scales[1].begin(35, 34);
  // scales[2].begin(33, 32);
  // scales[3].begin(29, 28);

  delay(1000);
  for (int i = 0; i < NUM_MOTORS; i++) {
    setpoints[i] = 300;
    Serial.println(i);
    loadCells[i].setScale(2.56);
    // scales[i].set_scale(432 * 5);
    motors[i].tare(&loadCells[i]);
    loadCells[i].tare();
    // scales[i].tare();
  }
}

void loop() {
  bool print = (counter++ % N_PRINT == 0);

  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    int spaceIndex1 = input.indexOf(' ');
    int spaceIndex2 = input.indexOf(' ', spaceIndex1 + 1);
    int spaceIndex3 = input.indexOf(' ', spaceIndex2 + 1);

    if (spaceIndex1 > 0 && spaceIndex2 > spaceIndex1 && spaceIndex3 > spaceIndex2) {
      setpoints[0] = input.substring(0, spaceIndex1).toInt();
      setpoints[1] = input.substring(spaceIndex1 + 1, spaceIndex2).toInt();
      setpoints[2] = input.substring(spaceIndex2 + 1, spaceIndex3).toInt();
      setpoints[3] = input.substring(spaceIndex3 + 1).toInt();
      if (print) Serial.println(setpoints[0]);
      if (print) Serial.println(setpoints[1]);
      if (print) Serial.println(setpoints[2]);
      if (print) Serial.println(setpoints[3]);
    }
  }

  currentTime = millis();
  float dt = (currentTime - previousTime) / 1000.0;
  previousTime = currentTime;

  for (int i = 0; i < NUM_MOTORS; i++) {
    readings[i] = loadCells[i].read();
    if (print) Serial.print(readings[i]);
    if (print) Serial.print(",");
    if (readings[i] > MAX_TENSION) {
      Serial.println("\nEMERGENCY STOP. EXCEEDED TENSION");
      // Release tension from all motors
      for (int i = 0; i < NUM_MOTORS; i++) motors[i].writePWM(-0.4);
      delay(500);
      for (int i = 0; i < NUM_MOTORS; i++) motors[i].writePWM(0);
      // Infinite loop so that no code ever executes
      while (true) { delay(1000); }
    }
  }

  for (int i = 0; i < NUM_MOTORS; i++) {
    float error = setpoints[i] - readings[i];
    float errorDerivative = (error - previousErrors[i]) / dt;
    integratedErrors[i] += error * dt;
    // integratedError * KI should be less than 1/2 of the output
    // to eliminate overshoot
    if (abs(integratedErrors[i]) > 255.0/(KI)) integratedErrors[i] = copysign(255.0/(KI), integratedErrors[i]);

    previousErrors[i] = error;
    float output = (KP * error + KI * integratedErrors[i] + KD * errorDerivative);
    // if (abs(error) < 20 && abs(errorDerivative) < 200) output = 0;
    // Serial.print(error);
    // Serial.print(",");
    // Serial.print(integratedErrors[i]);
    // Serial.print(",");
    // Serial.print(errorDerivative);
    // Serial.print(",");
    if (print) Serial.print(output);
    if (print) Serial.print(",");
    motors[i].writePWM(abs(output) < 0 ? 0 : output / 255.0);
  }
  accumDt += dt;
  if (print) {
    Serial.println(accumDt * 1000.0 / N_PRINT, 3);
    // Serial.println();
    accumDt = 0;
  }
  delay(1);
}
