#include <Arduino.h>
// #include "HX711.h"
#include "AnalogLoadCell.h"

/** A class for driving a servo in some calibrated range */
class TensionMotor {
  public:
  TensionMotor(uint8_t inA, uint8_t inB, uint8_t en);

  /** Initialize and tare */
  void tare(AnalogLoadCell *scale);

  /** Set using a tension in grams. **/
  void set(float tension, float dt);

  /** Set using a speed from -1 to 1. **/
  void writePWM(float speed);

  /** Turn motor off. **/
  void coast();

  float pwm; // Last pwm, for debugging

 private:
  uint8_t _inA;
  uint8_t _inB;
  uint8_t _en;
  float _lastTension;
  float _pGain;
  float _dGain;
  float _directionCorrection;
};
