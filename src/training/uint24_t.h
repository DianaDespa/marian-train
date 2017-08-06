#pragma once

#include <iostream>
using namespace std;

const int UINT24_MAX = 16777215;

class uint24_t
{
protected:
  unsigned char bytes[3];

public:
  uint24_t()
  {
  }

  uint24_t(const int val)
  {
    *this = val;
  }

  uint24_t(const uint24_t &val)
  {
    *this = val;
  }

  operator int() const
  {
    // never negative
    return (bytes[2] << 16) | (bytes[1] << 8) | (bytes[0] << 0);
  }

  operator float() const
  {
    return (float)this->operator int();
  }

  uint24_t &operator=(const uint24_t &input)
  {
    bytes[0] = input.bytes[0];
    bytes[1] = input.bytes[1];
    bytes[2] = input.bytes[2];

    return *this;
  }

  uint24_t &operator=(const int input)
  {
    bytes[0] = ((unsigned char *)&input)[0];
    bytes[1] = ((unsigned char *)&input)[1];
    bytes[2] = ((unsigned char *)&input)[2];

    return *this;
  }

  /***********************************************/

  uint24_t operator+(const uint24_t &val) const
  {
    return uint24_t((int)*this + (int)val);
  }

  uint24_t operator-(const uint24_t &val) const
  {
    return uint24_t((int)*this - (int)val);
  }

  uint24_t operator*(const uint24_t &val) const
  {
    return uint24_t((int)*this * (int)val);
  }

  uint24_t operator/(const uint24_t &val) const
  {
    return uint24_t((int)*this / (int)val);
  }

  /***********************************************/

  uint24_t operator+(const int val) const
  {
    return uint24_t((int)*this + val);
  }

  uint24_t operator-(const int val) const
  {
    return uint24_t((int)*this - val);
  }

  uint24_t operator*(const int val) const
  {
    return uint24_t((int)*this * val);
  }

  uint24_t operator/(const int val) const
  {
    return uint24_t((int)*this / val);
  }

  /***********************************************/
  /***********************************************/

  uint24_t &operator+=(const uint24_t &val)
  {
    *this = *this + val;
    return *this;
  }

  uint24_t &operator-=(const uint24_t &val)
  {
    *this = *this - val;
    return *this;
  }

  uint24_t &operator*=(const uint24_t &val)
  {
    *this = *this * val;
    return *this;
  }

  uint24_t &operator/=(const uint24_t &val)
  {
    *this = *this / val;
    return *this;
  }

  /***********************************************/

  uint24_t &operator+=(const int val)
  {
    *this = *this + val;
    return *this;
  }

  uint24_t &operator-=(const int val)
  {
    *this = *this - val;
    return *this;
  }

  uint24_t &operator*=(const int val)
  {
    *this = *this * val;
    return *this;
  }

  uint24_t &operator/=(const int val)
  {
    *this = *this / val;
    return *this;
  }

  /***********************************************/
  /***********************************************/

  uint24_t operator>>(const int val) const
  {
    return uint24_t((int)*this >> val);
  }

  uint24_t operator<<(const int val) const
  {
    return uint24_t((int)*this << val);
  }

  /***********************************************/

  uint24_t &operator>>=(const int val)
  {
    *this = *this >> val;
    return *this;
  }

  uint24_t &operator<<=(const int val)
  {
    *this = *this << val;
    return *this;
  }

  /***********************************************/
  /***********************************************/

  operator bool() const
  {
    return (int)*this != 0;
  }

  bool operator!() const
  {
    return !((int)*this);
  }

  uint24_t operator-()
  {
    return uint24_t(-(int)*this);
  }

  /***********************************************/
  /***********************************************/

  bool operator==(const uint24_t &val) const
  {
    return (int)*this == (int)val;
  }

  bool operator!=(const uint24_t &val) const
  {
    return (int)*this != (int)val;
  }

  bool operator>=(const uint24_t &val) const
  {
    return (int)*this >= (int)val;
  }

  bool operator<=(const uint24_t &val) const
  {
    return (int)*this <= (int)val;
  }

  bool operator>(const uint24_t &val) const
  {
    return (int)*this > (int)val;
  }

  bool operator<(const uint24_t &val) const
  {
    return (int)*this < (int)val;
  }

  /***********************************************/

  bool operator==(const int val) const
  {
    return (int)*this == val;
  }

  bool operator!=(const int val) const
  {
    return (int)*this != val;
  }

  bool operator>=(const int val) const
  {
    return (int)*this >= val;
  }

  bool operator<=(const int val) const
  {
    return (int)*this <= val;
  }

  bool operator>(const int val) const
  {
    return ((int)*this) > val;
  }

  bool operator<(const int val) const
  {
    return (int)*this < val;
  }

  /***********************************************/

  friend ostream& operator<<(ostream& os, const uint24_t &val);

  /***********************************************/
};

ostream& operator<<(ostream& os, const uint24_t &val)
{  
    os << (int)val;
    return os;  
}
