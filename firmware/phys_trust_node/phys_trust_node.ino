// PHYS-TRUST Lite node: MQ135 (A1) + DHT11 data (A0, used as digital pin). A2 = optional MQ7 (currently unconnected:
// mq7_adc is a floating-pin value, ignore it)
// -> serial CSV @115200, 1 Hz. Needs "DHT sensor library" (Adafruit) from Library Manager.
// Output: millis,adc_mean,adc_std,T,RH,dht_ok,mq7_adc
// MQ7 note: datasheet wants 5 V / 1.4 V heater cycling; plain module on constant 5 V
// gives only a rough CO-sensitive trend. Logged as auxiliary channel.
#include <DHT.h>

#define MQ135_PIN A1
#define MQ7_PIN A2
#define DHT_PIN A0
#define DHT_TYPE DHT11

const uint8_t N_OVERSAMPLE = 10;
const unsigned long SAMPLE_MS = 1000;
const unsigned long DHT_MS = 2000;  // DHT11 min period

DHT dht(DHT_PIN, DHT_TYPE);
unsigned long lastSample = 0, lastDht = 0;
float lastT = NAN, lastRH = NAN;
bool dhtOk = false;

void setup() {
  Serial.begin(115200);
  dht.begin();
  Serial.println(F("millis,adc_mean,adc_std,T,RH,dht_ok,mq7_adc"));
}

void loop() {
  unsigned long now = millis();

  if (now - lastDht >= DHT_MS) {
    lastDht = now;
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    dhtOk = !(isnan(t) || isnan(h));
    if (dhtOk) { lastT = t; lastRH = h; }
  }

  if (now - lastSample >= SAMPLE_MS) {
    lastSample = now;
    long sum = 0, sq = 0, sum7 = 0;
    analogRead(MQ135_PIN);  // discard: ADC mux settle (stops channel crosstalk)
    for (uint8_t i = 0; i < N_OVERSAMPLE; i++) {
      int v = analogRead(MQ135_PIN);
      sum += v;
      sq += (long)v * v;
      analogRead(MQ7_PIN);
      sum7 += analogRead(MQ7_PIN);
      analogRead(MQ135_PIN);
      delay(5);
    }
    float mean = (float)sum / N_OVERSAMPLE;
    float var = (float)sq / N_OVERSAMPLE - mean * mean;
    float sd = var > 0 ? sqrt(var) : 0;

    Serial.print(now); Serial.print(',');
    Serial.print(mean, 2); Serial.print(',');
    Serial.print(sd, 2); Serial.print(',');
    Serial.print(lastT, 1); Serial.print(',');
    Serial.print(lastRH, 1); Serial.print(',');
    Serial.print(dhtOk ? 1 : 0); Serial.print(',');
    Serial.println((float)sum7 / N_OVERSAMPLE, 2);
  }
}
