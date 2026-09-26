// Wiring diagnostic: per-pin ADC with discard read, DHT data line level, DHT11/DHT22 read.
#include <DHT.h>
DHT d11(A0, DHT11);
DHT d22(A0, DHT22);

int readClean(uint8_t pin) {
  analogRead(pin);            // discard: mux settles
  delay(2);
  return analogRead(pin);
}

void setup() {
  Serial.begin(115200);
  Serial.println(F("A0,A1,A2,A3,A4,A5,A1_digital,dht11_T,dht11_RH,dht22_T"));
}

void loop() {
  pinMode(A1, INPUT);
  int v[6];
  for (uint8_t i = 0; i < 6; i++) v[i] = readClean(A0 + i);
  int dig = digitalRead(A1);
  d11.begin();
  delay(1100);
  float t = d11.readTemperature(), h = d11.readHumidity();
  d22.begin();
  delay(1100);
  float t2 = d22.readTemperature();
  for (uint8_t i = 0; i < 6; i++) { Serial.print(v[i]); Serial.print(','); }
  Serial.print(dig); Serial.print(',');
  Serial.print(t); Serial.print(','); Serial.print(h); Serial.print(',');
  Serial.println(t2);
}
