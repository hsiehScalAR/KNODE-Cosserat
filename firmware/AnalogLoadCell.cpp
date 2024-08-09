#include "AnalogLoadCell.h"

AnalogLoadCell::AnalogLoadCell(uint8_t in) {
  _in = in;
}

void AnalogLoadCell::tare() {
  _offset = analogRead(_in);
}

void AnalogLoadCell::setScale(float scale) {
  _scale = scale;
}

float AnalogLoadCell::read() {
  return (analogRead(_in) - _offset) * _scale;
}
