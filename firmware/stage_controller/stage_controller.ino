// Arduino Nano + 2x TB6600 driver + 2x NEMA 17
// TB6600 DIP: OFF OFF OFF ON OFF ON → 32 microsteps, 6400 pulses/rev, 1A (1.2A peak)
// Serial command format:
//   "X 200\n"    move X axis 200 steps (negative = reverse)
//   "Y -500\n"   move Y axis -500 steps
//   "S 50\n"     set speed — stepDelay in µs (lower = faster, min ~10 µs)

// Motor X — TB6600 Driver 1
#define X_ENA 13
#define X_DIR 11
#define X_PUL 12

// Motor Y — TB6600 Driver 2
#define Y_ENA 5
#define Y_DIR 4
#define Y_PUL 3

#define STEPS_PER_REV 6400  // 200 full steps * 32 microsteps

// Pulse half-period in microseconds — controls speed.
// speed (rev/sec) = 1 / (2 * stepDelay_us * 1e-6 * STEPS_PER_REV)
// 78 µs → ~1 rev/sec  |  39 µs → ~2 rev/sec  |  156 µs → ~0.5 rev/sec
int stepDelay = 78;  // ~1 rev/sec default

void pulse(int pulPin) {
  digitalWrite(pulPin, HIGH);
  delayMicroseconds(stepDelay);
  digitalWrite(pulPin, LOW);
  delayMicroseconds(stepDelay);
}

void moveMotor(int enaPin, int dirPin, int pulPin, long steps) {
  digitalWrite(enaPin, LOW);                      // enable driver
  digitalWrite(dirPin, steps > 0 ? HIGH : LOW);   // set direction
  long absSteps = abs(steps);
  for (long i = 0; i < absSteps; i++) {
    pulse(pulPin);
  }
  digitalWrite(enaPin, HIGH);                     // disable after move (reduces heat)
}

void setup() {
  pinMode(X_ENA, OUTPUT); pinMode(X_DIR, OUTPUT); pinMode(X_PUL, OUTPUT);
  pinMode(Y_ENA, OUTPUT); pinMode(Y_DIR, OUTPUT); pinMode(Y_PUL, OUTPUT);

  digitalWrite(X_ENA, HIGH);  // start disabled
  digitalWrite(Y_ENA, HIGH);

  Serial.begin(2000000);
  Serial.println("Ready. Commands: 'X 200', 'Y -500' (steps), 'S 78' (speed delay µs)");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    char cmd = input.charAt(0);
    long value = input.substring(2).toInt();

    switch (cmd) {
      case 'X':
        moveMotor(X_ENA, X_DIR, X_PUL, value);
        Serial.println("X moved " + String(value) + " steps.");
        break;
      case 'Y':
        moveMotor(Y_ENA, Y_DIR, Y_PUL, value);
        Serial.println("Y moved " + String(value) + " steps.");
        break;
      case 'S':
        stepDelay = constrain((int)value, 10, 10000);
        Serial.println("Speed set: stepDelay=" + String(stepDelay) + "us (~" +
                       String(1000000L / (2L * stepDelay * STEPS_PER_REV)) + " rev/s)");
        break;
      default:
        Serial.println("Invalid. Use 'X 200', 'Y -500', or 'S 78'");
    }
  }
}
