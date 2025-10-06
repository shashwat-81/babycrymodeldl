/************* ESP32 MAX30102 + MLX90614 + Live Charts + Audio Upload/Record *************/
#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>

#include <Adafruit_MLX90614.h>
#include <MAX30105.h>
#include "heartRate.h"
#include "spo2_algorithm.h"

/* -------- Wi-Fi -------- */
const char* SSID     = "iPhone";
const char* PASSWORD = "12345678";

/* -------- Web server -------- */
WebServer server(80);

/* -------- Sensors -------- */
Adafruit_MLX90614 mlx;
MAX30105 sensor;

/* -------- Tunables -------- */
const byte IR_CURRENT   = 0x7F;
const byte RED_CURRENT  = 0x7F;
const byte FIFO_AVG     = 8;
const int  SAMPLE_RATE  = 25;
const int  PULSE_WIDTH  = 118;
const int  SAT_IR       = 250000;
const int  AC_MIN       = 100; // Try 100 or even 50
const int  WINDOW       = 64;

const int  BPM_MIN      = 40;
const int  BPM_MAX      = 180;
const float EMA_ALPHA   = 0.20f;

const int   SPO2_BUF = 100;
const int   MIN_AC_AMPL_SPO2 = 200;

/* -------- State -------- */
unsigned long lastBeatMs = 0;
float g_bpm = 0;
int   g_validBeats = 0;
bool  g_fingerOn = false;

float g_spo2 = 0;
int8_t g_spo2Valid = 0;

float g_Ta = NAN, g_To = NAN;

long ringIR[WINDOW];
int rp = 0;
bool ringFilled = false;

uint32_t irBuf[SPO2_BUF], redBuf[SPO2_BUF];
int bufIdx = 0;
bool bufFull = false;

static inline bool saturated(uint32_t ir) { return ir >= (uint32_t)SAT_IR; }
static inline int  pwToReg(int us){ if(us<=69)return 69; if(us<=118)return 118; if(us<=215)return 215; return 411; }

/* -------- HTML (wrapped in raw literal) -------- */
const char INDEX_HTML[] PROGMEM = R"HTML(
<!doctype html><html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESP32 Health Dashboard</title>
<style>
body{font-family:sans-serif;max-width:980px;margin:24px auto;padding:0 12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{border:1px solid #ddd;border-radius:12px;padding:14px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.badge{display:inline-block;padding:4px 8px;border-radius:999px;background:#f2f2f2;margin:4px}
button{padding:8px 12px;border:1px solid #ccc;border-radius:8px;background:#fff;cursor:pointer}
button.rec{background:#e91e63;color:#fff;border-color:#e91e63}
canvas{width:100%;height:240px}
</style>
</head><body>
<h1>ESP32 Health Dashboard</h1>
<div id="badges">
  <div class="badge" id="finger">no finger</div>
  <div class="badge" id="bpmNow">BPM: --</div>
  <div class="badge" id="spo2Now">SpO₂: --</div>
  <div class="badge" id="taNow">Ta: -- °C</div>
  <div class="badge" id="toNow">To: -- °C</div>
</div>
<div class="grid">
  <div class="card"><h3>BPM</h3><canvas id="bpmChart"></canvas></div>
  <div class="card"><h3>SpO₂ (%)</h3><canvas id="spo2Chart"></canvas></div>
  <div class="card"><h3>Ambient Temp (°C)</h3><canvas id="taChart"></canvas></div>
  <div class="card"><h3>Object/Skin Temp (°C)</h3><canvas id="toChart"></canvas></div>
</div>
<div class="card" style="margin-top:16px">
  <h3>Audio for Cry Classification</h3>
  <button id="recBtn" class="rec">● Record</button>
  <input id="fileInput" type="file" accept="audio/*">
  <button id="uploadBtn">Upload Selected</button>
  <div id="log" style="margin-top:8px;font-size:.9rem;color:#555;white-space:pre-line"></div>
  <audio id="player" controls style="margin-top:8px;max-width:100%"></audio>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
const maxPoints=120;
const bpmChart=makeLine('bpmChart');const spo2Chart=makeLine('spo2Chart');
const taChart=makeLine('taChart');const toChart=makeLine('toChart');
function makeLine(id){const ctx=document.getElementById(id).getContext('2d');
 return new Chart(ctx,{type:'line',data:{labels:[],datasets:[{label:id,borderWidth:2,pointRadius:0,data:[]}]},
 options:{animation:false,scales:{x:{display:false}}}});}
function pushPoint(chart,y){const t=new Date().toLocaleTimeString();
 chart.data.labels.push(t);chart.data.datasets[0].data.push(y);
 if(chart.data.labels.length>maxPoints){chart.data.labels.shift();chart.data.datasets[0].data.shift();}
 chart.update();}
async function poll(){try{const r=await fetch('/data');const d=await r.json();
 document.getElementById('finger').textContent=d.finger?'finger':'no finger';
 document.getElementById('bpmNow').textContent='BPM: '+(d.bpm??'--');
 document.getElementById('spo2Now').textContent='SpO₂: '+(d.spo2??'--');
 document.getElementById('taNow').textContent='Ta: '+(d.ta??'--')+' °C';
 document.getElementById('toNow').textContent='To: '+(d.to??'--')+' °C';
 if(d.bpm!==null)pushPoint(bpmChart,d.bpm);
 if(d.spo2!==null)pushPoint(spo2Chart,d.spo2);
 if(d.ta!==null)pushPoint(taChart,d.ta);
 if(d.to!==null)pushPoint(toChart,d.to);}catch(e){}}
poll();setInterval(poll,2000);
const recBtn=document.getElementById('recBtn');const fileInput=document.getElementById('fileInput');
const uploadBtn=document.getElementById('uploadBtn');const player=document.getElementById('player');
const log=(m)=>{document.getElementById('log').textContent=m;};
let mediaRecorder,chunks=[];
recBtn.onclick=async()=>{
 if(!mediaRecorder||mediaRecorder.state==='inactive'){
  try{const stream=await navigator.mediaDevices.getUserMedia({audio:true});
   mediaRecorder=new MediaRecorder(stream,{mimeType:'audio/webm'});chunks=[];
   mediaRecorder.ondataavailable=e=>{if(e.data.size>0)chunks.push(e.data);};
   mediaRecorder.onstop=async()=>{const blob=new Blob(chunks,{type:'audio/webm'});
    player.src=URL.createObjectURL(blob);await uploadBlob(blob,'recorded.webm');};
   mediaRecorder.start();recBtn.textContent='■ Stop';recBtn.classList.remove('rec');log('Recording…');
  }catch(err){log('Mic error:'+err);}
 }else{mediaRecorder.stop();recBtn.textContent='● Record';recBtn.classList.add('rec');log('Stopped. Uploading…');}
};
uploadBtn.onclick=async()=>{const f=fileInput.files[0];if(!f){log('Pick a file first.');return;}
 player.src=URL.createObjectURL(f);log('Uploading '+f.name+'…');await uploadBlob(f,f.name);};
async function uploadBlob(blob,name){const fd=new FormData();fd.append('audio',blob,name);
 const r=await fetch('/upload',{method:'POST',body:fd});const t=await r.text();log('Server:'+t);}
</script>
</body></html>
)HTML";

/* -------- Web Handlers -------- */
void handleRoot() { server.send_P(200, "text/html", INDEX_HTML); }

void handleData() {
  String json = "{";
  if (g_fingerOn && g_validBeats >= 3 && g_bpm > 0) json += "\"bpm\":" + String(g_bpm,1) + ","; else json += "\"bpm\":null,";
  if (g_fingerOn && g_spo2Valid) json += "\"spo2\":" + String(g_spo2,1) + ","; else json += "\"spo2\":null,";
  if (!isnan(g_Ta)) json += "\"ta\":" + String(g_Ta,1) + ","; else json += "\"ta\":null,";
  if (!isnan(g_To)) json += "\"to\":" + String(g_To,1) + ","; else json += "\"to\":null,";
  json += "\"finger\":" + String(g_fingerOn ? "true" : "false");
  json += "}";

  Serial.println(json); // <-- ADD THIS LINE

  server.send(200, "application/json", json);
}

// Upload handler
void handleUpload() {
  HTTPUpload& up = server.upload();
  static File f;
  if (up.status == UPLOAD_FILE_START) {
    String filename = "/" + (up.filename.length() ? up.filename : String("cry_last.webm"));
    if (SPIFFS.exists(filename)) SPIFFS.remove(filename);
    f = SPIFFS.open(filename, FILE_WRITE);
  } else if (up.status == UPLOAD_FILE_WRITE) {
    if (f) f.write(up.buf, up.currentSize);
  } else if (up.status == UPLOAD_FILE_END) {
    if (f) f.close();
  }
}
void handleUploadDone() { server.send(200, "text/plain", "Upload OK"); }

/* -------- Setup & Loop -------- */
void setup() {
  Serial.begin(115200);
  delay(150);
  if (!SPIFFS.begin(true)) Serial.println("SPIFFS mount failed");
  Wire.begin(21, 22);
  Wire.setClock(100000);
  delay(250);
  if (!mlx.begin()) Serial.println("MLX90614 NOT found"); else Serial.println("MLX90614 OK");
  if (!sensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 NOT found"); while(1) delay(10);
  }
  sensor.setup();
  sensor.setSampleRate(SAMPLE_RATE);
  sensor.setPulseWidth(pwToReg(PULSE_WIDTH));
  sensor.setPulseAmplitudeIR(IR_CURRENT);
  sensor.setPulseAmplitudeRed(RED_CURRENT);
  sensor.setPulseAmplitudeGreen(0);
  sensor.setFIFOAverage(FIFO_AVG);
  sensor.enableFIFORollover();

  WiFi.begin(SSID, PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.print("\nIP: "); Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.on("/upload", HTTP_POST, handleUploadDone, handleUpload);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  // read MAX30102
  if (sensor.safeCheck(10)) {
    while (sensor.available()) {
      long ir  = sensor.getIR();
      long red = sensor.getRed();
      sensor.nextSample();
      if (saturated(ir)) {
        g_fingerOn = false;
        g_bpm = 0; g_validBeats = 0; lastBeatMs = 0;
        ringIR[rp]=0;rp=(rp+1)%WINDOW;if(rp==0)ringFilled=true;
        irBuf[bufIdx]=0;redBuf[bufIdx]=0;bufIdx=(bufIdx+1)%SPO2_BUF;if(bufIdx==0)bufFull=true;
        continue;
      }
      ringIR[rp]=ir;rp=(rp+1)%WINDOW;if(rp==0)ringFilled=true;
      int n=ringFilled?WINDOW:rp;long vmin=ringIR[0],vmax=ringIR[0];
      for(int i=1;i<n;i++){if(ringIR[i]<vmin)vmin=ringIR[i];if(ringIR[i]>vmax)vmax=ringIR[i];}
      long ac=vmax-vmin;g_fingerOn=(ac>=AC_MIN);
      if(!g_fingerOn){g_bpm=0;g_validBeats=0;lastBeatMs=0;}else{
        if(checkForBeat(ir)){unsigned long now=millis();
          if(lastBeatMs>0){float dt=(now-lastBeatMs)/1000.0f;if(dt>0){
            float bpm=60.0f/dt;if(bpm>=BPM_MIN&&bpm<=BPM_MAX){
              g_bpm=(g_validBeats>0)?(g_bpm*(1.0f-EMA_ALPHA)+bpm*EMA_ALPHA):bpm;
              if(g_validBeats<20)g_validBeats++;}}}
          lastBeatMs=now;}
        if(lastBeatMs>0&&(millis()-lastBeatMs)>2000){g_bpm=0;g_validBeats=0;lastBeatMs=0;}
      }
      irBuf[bufIdx]=ir;redBuf[bufIdx]=red;bufIdx=(bufIdx+1)%SPO2_BUF;if(bufIdx==0)bufFull=true;
    }
  }
  // SpO2
  if(bufFull&&g_fingerOn){
    uint32_t irMin=irBuf[0],irMax=irBuf[0],redMin=redBuf[0],redMax=redBuf[0];
    for(int i=1;i<SPO2_BUF;i++){if(irBuf[i]<irMin)irMin=irBuf[i];if(irBuf[i]>irMax)irMax=irBuf[i];
      if(redBuf[i]<redMin)redMin=redBuf[i];if(redBuf[i]>redMax)redMax=redBuf[i];}
    int irAC=(int)(irMax-irMin);int redAC=(int)(redMax-redMin);
    if(irAC>MIN_AC_AMPL_SPO2&&redAC>MIN_AC_AMPL_SPO2){
      static uint32_t irTmp[SPO2_BUF],redTmp[SPO2_BUF];int start=bufIdx;
      for(int i=0;i<SPO2_BUF;i++){int k=(start+i)%SPO2_BUF;irTmp[i]=irBuf[k];redTmp[i]=redBuf[k];}
      int32_t dummyHR=0;int8_t dummyValid=0;
      maxim_heart_rate_and_oxygen_saturation(irTmp,SPO2_BUF,redTmp,&g_spo2,&g_spo2Valid,&dummyHR,&dummyValid);
    }else g_spo2Valid=0;
  }else if(!g_fingerOn)g_spo2Valid=0;
  // temps
  static unsigned long lastTemp=0;
  if(millis()-lastTemp>1000){lastTemp=millis();
    float Ta=mlx.readAmbientTempC();if(!isnan(Ta))g_Ta=Ta;
    float To=mlx.readObjectTempC();if(!isnan(To))g_To=To;}
  static unsigned long lastSerial = 0;
  if (millis() - lastSerial > 1000) { // every 1 second
    lastSerial = millis();
    handleData(); // send JSON to Serial
  }

  server.handleClient();
}
