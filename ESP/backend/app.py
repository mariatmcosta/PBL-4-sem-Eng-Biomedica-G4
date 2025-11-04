import serial
import time
import json
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import eventlet

# --- Configurações ---
# Ajuste 'COM3' para a porta serial correta do seu ESP32
# (ex: '/dev/ttyUSB0' no Linux/Mac ou 'COM3' no Windows)
SERIAL_PORT = '/dev/tty.Bluetooth-Incoming-Port' 
BAUD_RATE = 115200

# --- Configuração do Flask e SocketIO ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sua_chave_secreta_aqui' # Troque por uma chave segura
socketio = SocketIO(app, async_mode='eventlet')

# Variável para controlar o loop da thread serial
serial_thread_running = True

def read_serial_data():
    """Lê dados da porta serial e emite via SocketIO."""
    global serial_thread_running
    print(f"Tentando conectar ao ESP32 na porta {SERIAL_PORT}...")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Espera a conexão serial estabilizar
        print("Conexão serial estabelecida. Iniciando leitura de dados...")

        # Loop principal de leitura
        while serial_thread_running:
            if ser.in_waiting > 0:
                try:
                    # Lê a linha e decodifica para string
                    line = ser.readline().decode('utf-8').strip()
                    print(f"Dados recebidos do ESP32: {line}")

                    # Tenta converter a string JSON
                    # O ESP32 deve enviar os dados neste formato:
                    # {"imu1_roll": 1.23, "imu1_pitch": 4.56, "load_left": 50.1, "load_right": 49.9}
                    data_json = json.loads(line)
                    
                    # Emite os dados JSON para o frontend via SocketIO
                    socketio.emit('sensor_data', data_json, namespace='/test')
                    
                except json.JSONDecodeError:
                    print(f"Erro ao decodificar JSON: {line}")
                except Exception as e:
                    print(f"Erro durante a leitura/emissão: {e}")
            
            # Adiciona um pequeno delay para não sobrecarregar
            eventlet.sleep(0.01) 

    except serial.SerialException as e:
        print(f"Erro ao abrir a porta serial {SERIAL_PORT}: {e}")
        # Envia um erro para o frontend se não conseguir conectar
        socketio.emit('serial_error', {'message': str(e)}, namespace='/test')
    except Exception as e:
        print(f"Erro inesperado no thread serial: {e}")
        
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        print("Thread de leitura serial finalizada.")


@app.route('/')
def index():
    """Renderiza a página HTML principal."""
    # O HTML em 'templates/index.html' irá configurar o SocketIO
    return render_template('index.html')


@socketio.on('connect', namespace='/test')
def test_connect():
    """Evento disparado quando um cliente (navegador) se conecta via SocketIO."""
    print('Cliente conectado ao SocketIO.')
    emit('my response', {'data': 'Conectado ao servidor e aguardando dados.'})


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    """Evento disparado quando um cliente (navegador) se desconecta via SocketIO."""
    print('Cliente desconectado.')


# --- Inicialização ---
if __name__ == '__main__':
    # Inicia a thread que lida com a leitura serial separadamente
    serial_thread = threading.Thread(target=read_serial_data)
    serial_thread.daemon = True # Permite que o programa principal saia mesmo que a thread esteja rodando
    serial_thread.start()
    
    try:
        # Inicia o servidor SocketIO com o eventlet para operações assíncronas
        print("Servidor Flask e SocketIO rodando em http://127.0.0.1:5000/")
        socketio.run(app, debug=True, port=5000, host='0.0.0.0', use_reloader=False)

    except KeyboardInterrupt:
        print("Servidor desligado pelo usuário.")
    except Exception as e:
        print(f"Erro ao rodar o servidor: {e}")
    finally:
        # Garante que a thread serial pare ao desligar o servidor
        serial_thread_running = False
        serial_thread.join()
        print("Programa encerrado.")
