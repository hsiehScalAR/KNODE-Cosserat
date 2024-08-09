#include <Arduino.h>

/** A class for reading from an analog load cell */
class AnalogLoadCell {
  public:
  AnalogLoadCell(uint8_t input);

  /** Tare to zero the scale */
  void tare();

  /** Read the load cell value, in calibrated units */
  float read();

  /** Set the scale of the load cell */
  void setScale(float scale);

 private:
  uint8_t _in;
  float _scale = 1;
  float _offset = 0;
};
