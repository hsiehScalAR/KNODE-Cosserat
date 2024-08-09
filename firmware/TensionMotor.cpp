#include "TensionMotor.h"

#define TARE_THRESHOLD_SMALL 5
#define TARE_THRESHOLD_BIG 30

TensionMotor::TensionMotor(uint8_t inA, uint8_t inB, uint8_t en) {
  _inA = inA;
  _inB = inB;
  _en = en;
  _directionCorrection = -1;
}

void TensionMotor::tare(AnalogLoadCell *scale) {
  pinMode(_inA, OUTPUT);
  pinMode(_inB, OUTPUT);
  pinMode(_en, OUTPUT);
  // writePWM(-0.4);
  // delay(500);
  float previousValue = scale->read();
  while (true) {
    writePWM(0.2);
    delay(50);
    float newValue = scale->read();
    Serial.println(newValue - previousValue);

    // If tension increased significantly, proceed
    if (newValue > previousValue + TARE_THRESHOLD_BIG)
      break;
    // If tension decreased significantly, go the other direction
    // else if (newValue < previousValue - TARE_THRESHOLD_BIG) {
    //   Serial.println("Flip initial");
    //   _directionCorrection *= -1;
    // }


    Serial.println("Waiting for more tension");
    previousValue = newValue;
    // If tension stayed the same, keep going
  }

  while (true) {
    writePWM(-0.1);
    delay(50);
    float newValue = scale->read();
    // If the tension did not change significantly, stop taring.
    Serial.println(newValue - previousValue);
    if (abs(newValue - previousValue) < TARE_THRESHOLD_SMALL) {
      Serial.println("Stop taring");
      break;
    }

    Serial.println("Continue taring");
    previousValue = newValue;
    // If tension decrased, keep going
  }
  writePWM(0);
}

void TensionMotor::set(float tension, float dt) {
  float d_tension = (tension - _lastTension) / dt;
  _lastTension = tension;

  float desired = 0;

  float drive = _pGain * (desired - tension) + _dGain * (0 - d_tension);
  writePWM(drive);
}

void TensionMotor::writePWM(float speed) {
  pwm = speed;
  speed *= _directionCorrection;
  if (speed < -1) speed = -1;
  if (speed > 1) speed = 1;
  if (speed < 0) {
    digitalWrite(_inA, LOW);
    digitalWrite(_inB, HIGH);
    analogWrite(_en, 255 * -speed);
  } else if (speed > 0) {
    digitalWrite(_inA, HIGH);
    digitalWrite(_inB, LOW);
    analogWrite(_en, 255 * speed);
  } else {
    digitalWrite(_inA, LOW);
    digitalWrite(_inB, LOW);
    analogWrite(_en, 0);
  }
}

void TensionMotor::coast() {
    digitalWrite(_inA, HIGH);
    digitalWrite(_inB, HIGH);
}
