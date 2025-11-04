// Includes necessários para o ESP32
#include <Wire.h>           // Para comunicação I2C
#include <HX711_ADC.h>      // Para a célula de carga
#include <SparkFunMPU9250-DOF.h> // Para os IMUs

// --- HX711 ---
// Pinos 4 e 5 funcionam bem no ESP32
const int HX711_dout = 4;
const int HX711_sck  = 5;
HX711_ADC LoadCell(HX711_dout, HX711_sck);
float calibrationValue = 696.0; // Valor de calibração
unsigned long t = 0;

// --- IMUs ---
// Endereços I2C
#define IMU_ADDRESS1 0x68
#define IMU_ADDRESS2 0x69

MPU9250 IMU1; // Objeto IMU1
MPU9250 IMU2; // Objeto IMU2

void setup() {
  Serial.begin(57600); // Inicia a serial
  delay(100); // Pequeno atraso para garantir que a serial esteja pronta

  // --- HX711 Setup ---
  Serial.println("Iniciando HX711...");
  LoadCell.begin();
  LoadCell.startMultiple(2000, true); // Inicia em modo de alta velocidade
  LoadCell.setCalFactor(calibrationValue);
  Serial.println("Tara inicial... aguarde.");
  while(!LoadCell.getTareStatus()) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nTara concluída.");

  // --- IMU Setup ---
  Serial.println("Iniciando IMUs...");
  // No ESP32, os pinos I2C padrão são GPIO 21 (SDA) e GPIO 22 (SCL)
  Wire.begin(); 
  delay(1000); // Aguarda os sensores estabilizarem

  // CORREÇÃO: Usando 'IMU1' e 'IMU2' (maiúsculas) como declarado
  IMU1.begin(IMU_ADDRESS1);
  IMU2.begin(IMU_ADDRESS2);

  if (!IMU1.available()) Serial.println("IMU1 NÃO conectada!");
  else Serial.println("IMU1 ok (0x68)");

  if (!IMU2.available()) Serial.println("IMU2 NÃO conectada!");
  else Serial.println("IMU2 ok (0x69)");
  
  // Nota: A biblioteca SparkFunMPU9250-DOF pode não ter um 'testConnection()'
  // ou 'initialize()' da mesma forma que outras. '.begin()' e '.available()'
  // são os métodos mais comuns para ela.

  Serial.println("Sistema pronto.");
}

void loop() {
  static boolean newDataReady = 0;
  const int serialPrintInterval = 500; // Intervalo de 500ms

  // --- Atualiza HX711 ---
  if (LoadCell.update()) newDataReady = true;

  // --- Imprime dados se houver novos dados E o intervalo tiver passado ---
  if (newDataReady && (millis() - t > serialPrintInterval)) {
    float peso = LoadCell.getData();

    // --- Leitura IMU 1 ---
    // CORREÇÃO: Usando 'IMU1'
    if (IMU1.available()) {
      IMU1.update(); // Atualiza todos os dados do sensor

      Serial.print("IMU1 Acel: ");
      Serial.print(IMU1.getAccelX_mss(), 2); Serial.print(" ");
      Serial.print(IMU1.getAccelY_mss(), 2); Serial.print(" ");
      Serial.print(IMU1.getAccelZ_mss(), 2);

      Serial.print(" | Giro: ");
      Serial.print(IMU1.getGyroX_rads(), 2); Serial.print(" ");
      Serial.print(IMU1.getGyroY_rads(), 2); Serial.print(" ");
      Serial.println(IMU1.getGyroZ_rads(), 2);
    }

    // --- Leitura IMU 2 ---
    // CORREÇÃO: Usando 'IMU2'
    if (IMU2.available()) {
      IMU2.update(); // Atualiza todos os dados do sensor

      Serial.print("IMU2 Acel: ");
      Serial.print(IMU2.getAccelX_mss(), 2); Serial.print(" ");
      Serial.print(IMU2.getAccelY_mss(), 2); Serial.print(" ");
      Serial.print(IMU2.getAccelZ_mss(), 2);

      Serial.print(" | Giro: ");
      Serial.print(IMU2.getGyroX_rads(), 2); Serial.print(" ");
      Serial.print(IMU2.getGyroY_rads(), 2); Serial.print(" ");
      Serial.println(IMU2.getGyroZ_rads(), 2);
    }
    
    // --- PRINT Peso ---
    Serial.print("Peso: ");
    Serial.print(peso, 2); // Imprime com 2 casas decimais
    Serial.println(" g");
    
    Serial.println("---------------------");

    newDataReady = 0;
    t = millis();
  }

  // --- Bloco de Tara ---
  if (Serial.available() > 0) {
    char inByte = Serial.read();
    if (inByte == 't' || inByte == 'T') {
      LoadCell.tareNoDelay();
      Serial.println("Iniciando tara...");
    }
  }

  // Verifica se a tara foi concluída (no modo NoDelay)
  if (LoadCell.getTareStatus() == true) {
    Serial.println("Tara concluída!");
  }
}
