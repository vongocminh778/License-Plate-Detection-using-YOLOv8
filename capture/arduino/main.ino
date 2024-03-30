#include <Wire.h> 
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27,16,2); 
char receivedChar;

void setup() {
  Serial.begin(115200);
  lcd.init();                    
  lcd.backlight();
}

// Loop forever
void loop() {
  // If serial data is pending, read, capitalize and write the character
  if (Serial.available())
    lcd.clear();
    while (Serial.available()) {
      lcd.write(Serial.read());
    }
}