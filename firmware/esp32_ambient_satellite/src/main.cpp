#include <Arduino.h>
#include <ArduinoJson.h>
#include <Adafruit_NeoPixel.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <driver/i2s.h>
#include <math.h>
#include <string.h>

#if __has_include("secrets.h")
#include "secrets.h"
#else
#include "secrets.example.h"
#endif

namespace {

constexpr gpio_num_t PIR_PIN = GPIO_NUM_13;
constexpr gpio_num_t MIC_BCLK_PIN = GPIO_NUM_14;
constexpr gpio_num_t MIC_WS_PIN = GPIO_NUM_15;
constexpr gpio_num_t MIC_DATA_PIN = GPIO_NUM_32;
constexpr gpio_num_t SPEAKER_BCLK_PIN = GPIO_NUM_26;
constexpr gpio_num_t SPEAKER_LRCLK_PIN = GPIO_NUM_16;
constexpr gpio_num_t SPEAKER_DATA_PIN = GPIO_NUM_17;
constexpr uint8_t STATUS_LED_PIN = 21;
constexpr uint16_t STATUS_LED_COUNT = 1;
constexpr uint32_t INPUT_SAMPLE_RATE = 16000;
constexpr uint32_t OUTPUT_SAMPLE_RATE = 24000;
constexpr size_t AUDIO_SAMPLES_PER_FRAME = 256;
constexpr uint32_t WAKE_RMS_THRESHOLD = 2600;
constexpr uint32_t WAKE_REARM_MS = 3000;
constexpr uint32_t WIFI_RETRY_MS = 10000;

WebSocketsClient webSocket;
Adafruit_NeoPixel statusLed(
    STATUS_LED_COUNT, STATUS_LED_PIN, NEO_GRB + NEO_KHZ800
);
volatile bool pirChanged = false;
volatile bool pirLevel = false;
bool socketConnected = false;
bool socketStarted = false;
bool hardwareReady = false;
bool backendMicActive = false;
bool wakeRequestPending = false;
uint32_t lastWakeRequestMs = 0;
uint32_t lastWifiAttemptMs = 0;
uint32_t lastTelemetryPirLevelMs = 0;
String extraHeaders;

void IRAM_ATTR onPirEdge() {
  pirChanged = true;
}

bool configurationReady() {
  return strlen(FRIDAY_WIFI_SSID) > 0 &&
         strlen(FRIDAY_WIFI_PASSWORD) > 0 &&
         strlen(FRIDAY_WS_HOST) > 0 &&
         strlen(FRIDAY_API_TOKEN) >= 32 &&
         strlen(FRIDAY_WS_CA_CERT) > 0 &&
         strstr(FRIDAY_WS_CA_CERT, "BEGIN CERTIFICATE") != nullptr;
}

void setStatusColor(uint8_t red, uint8_t green, uint8_t blue) {
  statusLed.setPixelColor(0, statusLed.Color(red, green, blue));
  statusLed.show();
}

void updateStatusLed() {
  if (!socketConnected || !backendMicActive) {
    setStatusColor(0, 0, 0);
    return;
  }
  setStatusColor(181, 89, 51);
}

void sendControl(const JsonDocument& message) {
  if (!socketConnected) {
    return;
  }
  String serialized;
  serializeJson(message, serialized);
  webSocket.sendTXT(serialized);
}

void sendTelemetry(bool motion) {
  JsonDocument message;
  message["type"] = "telemetry";
  message["pir_active"] = motion;
  sendControl(message);
}

void sendWakeRequest() {
  JsonDocument message;
  message["type"] = "wake_word";
  message["source"] = "energy_threshold";
  sendControl(message);
  wakeRequestPending = true;
  lastWakeRequestMs = millis();
}

bool installAudioPipelines() {
  i2s_config_t microphoneConfig = {};
  microphoneConfig.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
  microphoneConfig.sample_rate = INPUT_SAMPLE_RATE;
  microphoneConfig.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  microphoneConfig.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  microphoneConfig.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  microphoneConfig.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  microphoneConfig.dma_buf_count = 8;
  microphoneConfig.dma_buf_len = 256;
  microphoneConfig.use_apll = false;
  microphoneConfig.tx_desc_auto_clear = false;
  microphoneConfig.fixed_mclk = 0;
  i2s_pin_config_t microphonePins = {};
  microphonePins.bck_io_num = MIC_BCLK_PIN;
  microphonePins.ws_io_num = MIC_WS_PIN;
  microphonePins.data_out_num = I2S_PIN_NO_CHANGE;
  microphonePins.data_in_num = MIC_DATA_PIN;
  if (i2s_driver_install(I2S_NUM_0, &microphoneConfig, 0, nullptr) != ESP_OK ||
      i2s_set_pin(I2S_NUM_0, &microphonePins) != ESP_OK) {
    return false;
  }

  i2s_config_t speakerConfig = {};
  speakerConfig.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_TX);
  speakerConfig.sample_rate = OUTPUT_SAMPLE_RATE;
  speakerConfig.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  speakerConfig.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
  speakerConfig.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  speakerConfig.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  speakerConfig.dma_buf_count = 8;
  speakerConfig.dma_buf_len = 256;
  speakerConfig.use_apll = false;
  speakerConfig.tx_desc_auto_clear = true;
  speakerConfig.fixed_mclk = 0;
  i2s_pin_config_t speakerPins = {};
  speakerPins.bck_io_num = SPEAKER_BCLK_PIN;
  speakerPins.ws_io_num = SPEAKER_LRCLK_PIN;
  speakerPins.data_out_num = SPEAKER_DATA_PIN;
  speakerPins.data_in_num = I2S_PIN_NO_CHANGE;
  return i2s_driver_install(I2S_NUM_1, &speakerConfig, 0, nullptr) == ESP_OK &&
         i2s_set_pin(I2S_NUM_1, &speakerPins) == ESP_OK;
}

uint32_t calculateRms(const int16_t* samples, size_t count) {
  if (count == 0) {
    return 0;
  }
  uint64_t sumSquares = 0;
  for (size_t index = 0; index < count; ++index) {
    const int32_t sample = samples[index];
    sumSquares += static_cast<uint64_t>(sample * sample);
  }
  return static_cast<uint32_t>(sqrt(
      static_cast<double>(sumSquares) / static_cast<double>(count)
  ));
}

void streamMicrophoneFrame(const int16_t* samples, size_t count) {
  if (!socketConnected || !backendMicActive || count == 0) {
    return;
  }
  webSocket.sendBIN(
      reinterpret_cast<uint8_t*>(const_cast<int16_t*>(samples)),
      count * sizeof(int16_t)
  );
}

void playSpeakerFrame(const uint8_t* payload, size_t length) {
  if (payload == nullptr || length < sizeof(int16_t) ||
      length % sizeof(int16_t) != 0) {
    return;
  }
  constexpr size_t INPUT_SAMPLES_PER_WRITE = 256;
  int16_t stereo[INPUT_SAMPLES_PER_WRITE * 2];
  const size_t totalSamples = length / sizeof(int16_t);
  size_t offset = 0;
  while (offset < totalSamples) {
    const size_t samplesThisWrite =
        min(INPUT_SAMPLES_PER_WRITE, totalSamples - offset);
    for (size_t index = 0; index < samplesThisWrite; ++index) {
      const size_t byteOffset = (offset + index) * sizeof(int16_t);
      const uint16_t raw = static_cast<uint16_t>(payload[byteOffset]) |
                           (static_cast<uint16_t>(payload[byteOffset + 1]) << 8);
      const int16_t sample = static_cast<int16_t>(raw);
      stereo[index * 2] = sample;
      stereo[index * 2 + 1] = sample;
    }
    size_t written = 0;
    const esp_err_t result = i2s_write(
        I2S_NUM_1,
        stereo,
        samplesThisWrite * 2 * sizeof(int16_t),
        &written,
        portMAX_DELAY
    );
    if (result != ESP_OK || written == 0) {
      Serial.println("Speaker I2S write failed.");
      return;
    }
    offset += samplesThisWrite;
  }
}

void handleControlMessage(uint8_t* payload, size_t length) {
  JsonDocument message;
  const DeserializationError error = deserializeJson(message, payload, length);
  if (error) {
    Serial.printf("Ignoring malformed backend control message: %s\n", error.c_str());
    return;
  }

  const char* type = message["type"] | "";
  if (strcmp(type, "ready") == 0 ||
      strcmp(type, "presence_state") == 0 ||
      strcmp(type, "telemetry_ack") == 0) {
    backendMicActive = message["mic_active"] | false;
    wakeRequestPending = false;
    updateStatusLed();
  } else if (strcmp(type, "audio_discarded") == 0) {
    backendMicActive = false;
    wakeRequestPending = false;
    updateStatusLed();
  } else if (strcmp(type, "error") == 0) {
    Serial.printf("Backend rejected satellite control: %s\n",
                  message["detail"] | "unspecified error");
  }
}

void onWebSocketEvent(
    WStype_t eventType, uint8_t* payload, size_t length
) {
  switch (eventType) {
    case WStype_CONNECTED: {
      socketConnected = true;
      backendMicActive = false;
      wakeRequestPending = false;
      JsonDocument handshake;
      handshake["client"] = "esp32_satellite";
      handshake["sample_rate"] = INPUT_SAMPLE_RATE;
      handshake["channels"] = 1;
      handshake["format"] = "s16le";
      sendControl(handshake);
      sendTelemetry(digitalRead(PIR_PIN) == HIGH);
      updateStatusLed();
      break;
    }
    case WStype_DISCONNECTED:
      socketConnected = false;
      backendMicActive = false;
      wakeRequestPending = false;
      updateStatusLed();
      break;
    case WStype_TEXT:
      handleControlMessage(payload, length);
      break;
    case WStype_BIN:
      setStatusColor(0, 190, 210);
      playSpeakerFrame(payload, length);
      updateStatusLed();
      break;
    case WStype_ERROR:
      socketConnected = false;
      backendMicActive = false;
      updateStatusLed();
      Serial.println("Secure WebSocket transport reported an error.");
      break;
    default:
      break;
  }
}

void maintainWifi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }
  const uint32_t now = millis();
  if (now - lastWifiAttemptMs < WIFI_RETRY_MS) {
    return;
  }
  lastWifiAttemptMs = now;
  WiFi.disconnect();
  WiFi.begin(FRIDAY_WIFI_SSID, FRIDAY_WIFI_PASSWORD);
}

void processMicrophone() {
  int32_t rawSamples[AUDIO_SAMPLES_PER_FRAME];
  size_t bytesRead = 0;
  const esp_err_t result = i2s_read(
      I2S_NUM_0,
      rawSamples,
      sizeof(rawSamples),
      &bytesRead,
      pdMS_TO_TICKS(40)
  );
  if (result != ESP_OK || bytesRead < sizeof(int32_t)) {
    return;
  }

  int16_t pcmSamples[AUDIO_SAMPLES_PER_FRAME];
  const size_t sampleCount = bytesRead / sizeof(int32_t);
  for (size_t index = 0; index < sampleCount; ++index) {
    const int32_t shifted = rawSamples[index] >> 14;
    pcmSamples[index] = static_cast<int16_t>(
        constrain(shifted, INT16_MIN, INT16_MAX)
    );
  }

  if (backendMicActive) {
    streamMicrophoneFrame(pcmSamples, sampleCount);
  } else if (socketConnected && !wakeRequestPending) {
    const uint32_t now = millis();
    if (now - lastWakeRequestMs >= WAKE_REARM_MS &&
        calculateRms(pcmSamples, sampleCount) >= WAKE_RMS_THRESHOLD) {
      sendWakeRequest();
    }
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  pinMode(PIR_PIN, INPUT);
  pirLevel = digitalRead(PIR_PIN) == HIGH;
  attachInterrupt(digitalPinToInterrupt(PIR_PIN), onPirEdge, CHANGE);
  statusLed.begin();
  statusLed.setBrightness(48);
  setStatusColor(0, 0, 0);

  if (!configurationReady()) {
    Serial.println(
        "Satellite locked: configure Wi-Fi, API token, and a valid TLS CA certificate."
    );
    return;
  }
  if (!installAudioPipelines()) {
    Serial.println("Satellite locked: I2S initialization failed.");
    return;
  }
  hardwareReady = true;

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(FRIDAY_WIFI_SSID, FRIDAY_WIFI_PASSWORD);
  lastWifiAttemptMs = millis();

  extraHeaders = String("Authorization: Bearer ") + FRIDAY_API_TOKEN + "\r\n";
  webSocket.setExtraHeaders(extraHeaders.c_str());
  webSocket.onEvent(onWebSocketEvent);
  webSocket.setReconnectInterval(5000);
  webSocket.enableHeartbeat(15000, 3000, 2);
}

void loop() {
  if (!configurationReady() || !hardwareReady) {
    delay(250);
    return;
  }
  maintainWifi();
  if (WiFi.status() == WL_CONNECTED) {
    if (!socketStarted) {
      webSocket.beginSslWithCA(
          FRIDAY_WS_HOST,
          FRIDAY_WS_PORT,
          FRIDAY_WS_PATH,
          FRIDAY_WS_CA_CERT,
          "friday-satellite"
      );
      socketStarted = true;
    }
    webSocket.loop();
  } else {
    socketStarted = false;
  }

  bool motionChanged = false;
  bool motion = false;
  noInterrupts();
  if (pirChanged) {
    motionChanged = true;
    motion = digitalRead(PIR_PIN) == HIGH;
    pirLevel = motion;
    pirChanged = false;
  }
  interrupts();
  if (motionChanged && socketConnected) {
    sendTelemetry(motion);
    lastTelemetryPirLevelMs = millis();
  }

  if (socketConnected) {
    const bool currentMotion = digitalRead(PIR_PIN) == HIGH;
    if (currentMotion != pirLevel &&
        millis() - lastTelemetryPirLevelMs >= 100) {
      pirLevel = currentMotion;
      sendTelemetry(currentMotion);
      lastTelemetryPirLevelMs = millis();
    }
  }

  processMicrophone();
  delay(1);
}
